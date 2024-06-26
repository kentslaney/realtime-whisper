import numpy as np
import asyncio
import torch

from whisper.audio import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_FRAMES,
    mel_filters,
)

# https://stackoverflow.com/a/17511341/3476782
def ceildiv(a, b):
    return -(a // -b)

async def apeek(iterator):
    try:
        initial = await anext(iterator)
    except StopAsyncIteration:
        return iter(()), (), None
    async def concat():
        yield initial
        async for v in iterator:
            yield v
    return concat(), initial.shape, initial.dtype

async def resample(iterator, size, axis=-1, exact=False):
    clean, init = True, 0
    iterator, shape, dtype = await apeek(iterator)
    if dtype is None:
        return
    axis = len(shape) + axis if axis < 0 else axis
    take = lambda sample, pos: sample[(slice(None),) * axis + (pos,)]
    empty, concat = (np.empty, np.concatenate) if isinstance(dtype, np.dtype) \
            else (torch.zeros, torch.cat)
    reset = running = empty((0,) * len(shape), dtype=dtype)
    async for sample in iterator:
        if not clean and running.shape[axis] > 0:
            if sample.shape[axis] < init:
                init -= sample.shape[axis]
                running = concat((running, sample), axis)
                continue
            ending = take(sample, slice(0, init))
            yield concat((running, ending), axis)
            running = reset
        remainder = (sample.shape[axis] - init) % size
        if exact:
            bounds = [range(
                    init, sample.shape[axis] + i, size)[i:] for i in range(2)]
            for pos in map(slice, *bounds):
                yield take(sample, pos)
        else:
            end = sample.shape[axis] - remainder
            if init != end:
                yield take(sample, slice(init, end))
        if remainder > 0:
            clean = False
            running = take(sample, slice(-remainder, None))
        init = remainder and size - remainder
    if running.shape[axis] > 0:
        yield running

class AudioSink:
    def __init__(self, *, rate=SAMPLE_RATE, **kw):
        super().__init__(**kw)
        self.rate = rate

    def read():
        raise NotImplementedError

    def write(data):
        raise NotImplementedError

class ArrayStream(AudioSink):
    def __init__(
            self, *, device=None, batch=1, n_mels=80, capacity=1_000_000, **kw):
        super().__init__(**kw)
        self.q = asyncio.Queue(capacity)
        self.finished = asyncio.Event()
        self.device, self.batch, self.n_mels = device, batch, n_mels
        self.kw = {"dtype": torch.float32, "device": self.device}
        self.sees = torch.zeros((0,), **self.kw)
        self.spectogram = torch.zeros((n_mels, 0), **self.kw)
        self.hann = torch.hann_window(N_FFT).to(self.sees.device)
        self.filters = mel_filters(self.sees.device, n_mels)

    def write(self, data):
        self.q.put_nowait(data)

    def load(self, data):
        return np.frombuffer(
                data, np.int16).flatten().astype(np.float32) / 32768.0

    async def loader(self, iterator):
        async for data in iterator:
            yield self.load(data)

    async def buffer(self):
        waiter = asyncio.create_task(self.finished.wait())
        while not self.finished.is_set():
            getter = asyncio.create_task(self.q.get())
            done, pending = await asyncio.wait(
                    (waiter, getter), return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield getter.result()
        while not self.q.empty():
            yield self.q.get_nowait()

    async def buffer_nowait(self):
        try:
            while True:
                yield self.q.get_nowait()
        except asyncio.QueueEmpty:
            pass

    async def fft_offset(self, iterator):
        iterator = resample(self.loader(iterator), HOP_LENGTH)
        window = np.zeros((0,), dtype=np.float32)
        while window.size < ceildiv(N_FFT, 2):
            try:
                window = np.concatenate((window, await anext(iterator)))
            except StopAsyncIteration:
                return
        window = np.pad(window, (N_FFT // 2, 0), 'reflect')
        yield window
        async for data in iterator:
            yield data
        # for _ in range(N_FFT // HOP_LENGTH):
        #     yield np.zeros((HOP_LENGTH,), dtype=np.float32)
        # (done by runoff)

    def seeing(self, sees):
        hopped = ((sees.shape[0] - N_FFT) // HOP_LENGTH + 1) * HOP_LENGTH
        return sees[hopped:]

    async def window(self, iterator):
        # bit ironic referring to the (in-buffer) audio data as "seen"
        iterator = self.fft_offset(iterator)
        async for data in iterator:
            data = torch.from_numpy(data)
            prev = self.sees.shape[0] - N_FFT
            while (data.shape[0] + prev) // HOP_LENGTH < self.batch - 1:
                try:
                    adding = torch.from_numpy(await anext(iterator))
                except StopAsyncIteration:
                    break
                data = torch.cat((data, adding))
            if self.device is not None:
                data.to(self.device)
            res = torch.cat((self.sees, data))
            self.sees = self.seeing(res)
            yield self.transform(self.dft(res))

    def dft(self, amp):
        return torch.stft(
                amp, N_FFT, HOP_LENGTH, window=self.hann, center=False,
                return_complex=True)

    # https://github.com/openai/whisper/blob/c5d4256/whisper/audio.py#L149
    def transform(self, stft):
        magnitudes = stft.abs() ** 2
        mel_spec = self.filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def padding(self, content_frames):
        return N_FRAMES

    def runoff(self):
        overrun = (ceildiv(N_FFT, HOP_LENGTH) - 1) * HOP_LENGTH
        spectogram = torch.cat((self.sees, torch.zeros(overrun, **self.kw)))
        if spectogram.shape[-1] >= N_FFT:
            spectogram = self.transform(self.dft(spectogram))
        padding = self.padding(self.spectogram.shape[-1] + spectogram.shape[-1])
        pad = torch.zeros(self.n_mels, max(0, padding), **self.kw)
        spectogram = torch.cat((self.spectogram, spectogram, pad), -1)
        return spectogram if padding >= 0 else spectogram[-padding:]

    offset = 0

    async def pull(self):
        context = self.spectogram.shape[-1]
        iterator = self.window(self.buffer_nowait())
        async for frame in iterator:
            self.spectogram = torch.cat((self.spectogram, frame), -1)
        cutoff = min(context, max(self.spectogram.shape[-1] - N_FRAMES, 0))
        self.offset += cutoff
        self.spectogram = self.spectogram[:, cutoff:]
        return self.runoff()

    async def _push(self, sec, exact=False):
        resampling = sec * SAMPLE_RATE // HOP_LENGTH
        iterator = self.window(self.buffer())
        async for frame in resample(iterator, resampling, exact=exact):
            resampled = resampling if exact else frame.shape[-1]
            cutoff = max(self.spectogram.shape[-1] + resampled - N_FRAMES, 0)
            self.offset += cutoff
            self.spectogram = torch.cat((
                    self.spectogram[:, cutoff:], frame), -1)
            yield self.runoff()

    async def push(self, sec, exact=False, **kw):
        reader = asyncio.create_task(self.read(**kw))
        async for i in self._push(sec, exact):
            yield i
        await reader

    async def full(self, **kw):
        await self.read(**kw)
        return await self.pull()

    def sequential(self, **kw):
        asyncio.run(self.read(**kw))
        return asyncio.run(self.pull())

class RawAudioFile(ArrayStream):
    def __init__(self, *, period=HOP_LENGTH, fname='out.raw', **kw):
        super().__init__(**kw)
        self.fname = fname
        self.period = period

    async def read(self):
        fp = open(self.fname, 'rb')
        data = fp.read(self.period)
        while len(data) != 0:
            self.write(data)
            data = fp.read(self.period)
        self.finished.set()

class AudioFile(RawAudioFile):
    def __init__(self, *, period=SAMPLE_RATE, fname='out.wav', **kw):
        global subprocess
        import subprocess
        assert not subprocess.run(
            ["which", "ffmpeg"], stdout=subprocess.PIPE).returncode
        super().__init__(period=period or -1, fname=fname, **kw)

    async def read(self):
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", self.fname,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(self.rate),
            "-"
        ]
        ps = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.fp = ps.stdout
        await super().read()
        if ps.returncode not in (None, 0):
            raise Exception

class SequenceDone(Exception):
    pass

class AudioFileSequence(AudioFile):
    def __init__(self, *, seq='*.wav', **kw):
        global glob
        import glob
        super().__init__(fname=None, **kw)
        seq = (seq,) if isinstance(seq, str) else seq
        self.seq = tuple(j for i in seq for j in glob.glob(i))
        self.idx = 0

    def clear(self):
        self.finished.clear()

    def read(self):
        async def inner():
            if self.idx >= len(self.seq):
                raise SequenceDone
            self.fname = self.seq[self.idx]
            await super(__class__, self).read()
            self.idx += 1
        self.clear()
        return inner()

class AudioFileStitch(ArrayStream):
    def __init__(self, *, seq="*.wav", period=SAMPLE_RATE, **kw):
        super().__init__(**kw)
        self.seq = AudioFileSequence(seq=seq, period=period, **kw)
        self.seq.q = self.q

    async def read(self):
        try:
            while True:
                await self.seq.read()
        except SequenceDone:
            self.finished.set()

class LiveCapture(ArrayStream):
    def __init__(
            self, *, period=HOP_LENGTH, source='default', loop_delay=.001,
            **kw):
        global alsaaudio
        import alsaaudio
        super().__init__(**kw)
        self.period, self.source, self.loop_delay = period, source, loop_delay

    async def busy_loop(self):
        inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK,
            channels=1, rate=self.rate, format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=self.period, device=self.source)

        while True:
            l, data = inp.read()
            if l < 0:
                raise Exception("capture buffer overrun before write buffer")
            if l:
                self.write(data)
                await asyncio.sleep(self.loop_delay)

    async def read(self, length=None):
        if length is None:
            await self.busy_loop()
        try:
            async with asyncio.timeout(length):
                await self.busy_loop()
        except TimeoutError:
            self.finished.set()

    def blocking(self, length=None):
        asyncio.run(self.read(length))

class Recorder(LiveCapture):
    def __init__(self, *, fname='out.raw', **kw):
        super().__init__(**kw)
        self.fp = open(fname, 'wb')

    def write(self, data):
        self.fp.write(data)


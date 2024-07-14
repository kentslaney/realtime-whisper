import numpy as np
import asyncio, pathlib, subprocess, torch

from whisper.audio import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_FRAMES,
    mel_filters,
)

from utils import Batcher, PathType, ceildiv
from typing import Optional, Union, IO, Tuple, Any, Type
from collections.abc import Coroutine, AsyncIterable, AsyncIterator, Awaitable

class AudioSink:
    def __init__(self, *, rate: int = SAMPLE_RATE, **kw):
        super().__init__(**kw)
        self.rate = rate

    def read(self):
        raise NotImplementedError

    def write(self, data):
        raise NotImplementedError

class ArrayStream(AudioSink):
    q: asyncio.Queue
    def __init__(
            self, *, device: Optional[Union[str, torch.device]] = None,
            batch: int = 1, n_mels: int = 80, capacity: int = 1_000_000, **kw):
        super().__init__(**kw)
        self.q = asyncio.Queue(capacity)
        self.finished = asyncio.Event()
        self.device, self.batch, self.n_mels = device, batch, n_mels
        self.sees = self.zeros((0,))
        self.spectogram = self.zeros((n_mels, 0))
        self.hann = torch.hann_window(N_FFT).to(self.sees.device)
        self.filters = mel_filters(self.sees.device, n_mels)

    def zeros(self, shape):
        return torch.zeros(shape, dtype=torch.float32, device=self.device)

    write_blockable: bool = True
    def write(self, data: bytes) -> Optional[Coroutine]:
        if self.write_blockable:
            return self.q.put(data)
        else:
            self.q.put_nowait(data)
            return None

    def load(self, data: bytes) -> np.ndarray:
        return np.frombuffer(
                data, np.int16).flatten().astype(np.float32) / 32768.0

    async def loader(self, iterator: AsyncIterable[bytes]) -> \
            AsyncIterator[np.ndarray]:
        async for data in iterator:
            yield self.load(data)

    async def buffer(self) -> AsyncIterator[bytes]:
        waiter = asyncio.create_task(self.finished.wait())
        while not self.finished.is_set():
            getter = asyncio.create_task(self.q.get())
            done, pending = await asyncio.wait(
                    (waiter, getter), return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield getter.result()
        while not self.q.empty():
            yield self.q.get_nowait()

    async def buffer_nowait(self) -> AsyncIterator[bytes]:
        try:
            while True:
                yield self.q.get_nowait()
        except asyncio.QueueEmpty:
            pass

    loading: Optional[Batcher] = None
    async def fft_offset(self, iterator: AsyncIterable[bytes]) -> \
            AsyncIterator[np.ndarray]:
        init = self.loader(iterator) if self.loading is None else self.loading
        self.loading = Batcher(init, HOP_LENGTH)
        _iterator = aiter(self.loading)
        window = np.zeros((0,), dtype=np.float32)
        while window.size < ceildiv(N_FFT, 2):
            try:
                window = np.concatenate((window, await anext(_iterator)))
            except StopAsyncIteration:
                return
        window = np.pad(window, (N_FFT // 2, 0), 'reflect')
        yield window
        async for data in _iterator:
            yield data
        # for _ in range(N_FFT // HOP_LENGTH):
        #     yield np.zeros((HOP_LENGTH,), dtype=np.float32)
        # (done by runoff)

    def seeing(self, sees: torch.Tensor) -> torch.Tensor:
        hopped = ((sees.shape[0] - N_FFT) // HOP_LENGTH + 1) * HOP_LENGTH
        return sees[hopped:]

    async def window(self, iterator: AsyncIterable[bytes]) -> \
            AsyncIterator[torch.Tensor]:
        _iterator = self.fft_offset(iterator)
        async for data in _iterator:
            _data = torch.from_numpy(data)
            prev = self.sees.shape[0] - N_FFT
            while (_data.shape[0] + prev) // HOP_LENGTH < self.batch - 1:
                try:
                    adding = torch.from_numpy(await anext(_iterator))
                except StopAsyncIteration:
                    break
                _data = torch.cat((_data, adding))
            if self.device is not None:
                _data.to(self.device)
            res = torch.cat((self.sees, _data))
            self.sees = self.seeing(res)
            yield self.transform(self.dft(res))

    def dft(self, amp: torch.Tensor) -> torch.Tensor:
        return torch.stft(
                amp, N_FFT, HOP_LENGTH, window=self.hann, center=False,
                return_complex=True)

    # https://github.com/openai/whisper/blob/c5d4256/whisper/audio.py#L149
    log_spec_bound: Optional[torch.Tensor] = None
    def transform(self, stft: torch.Tensor) -> torch.Tensor:
        magnitudes = stft.abs() ** 2
        mel_spec = self.filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        # causes values to not precisely match the original
        self.log_spec_bound = log_spec.max() if self.log_spec_bound is None \
                else torch.maximum(log_spec.max(), self.log_spec_bound)
        log_spec = torch.maximum(log_spec, self.log_spec_bound - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def padding(self, content_frames: int) -> int:
        return N_FRAMES

    # dft_pad: add ending content frames to match padding from a centered STFT
    dft_pad: bool = False
    def runoff(self, dft_pad: Optional[bool] = None) -> torch.Tensor:
        dft_pad = self.dft_pad if dft_pad is None else dft_pad
        if dft_pad:
            overrun = (ceildiv(N_FFT, HOP_LENGTH) - 1) * HOP_LENGTH
            spectogram = torch.cat((self.sees, self.zeros(overrun)))
            if spectogram.shape[-1] >= N_FFT:
                spectogram = self.transform(self.dft(spectogram))
        else:
            spectogram = torch.zeros(0)
        padding = self.padding(self.spectogram.shape[-1] + spectogram.shape[-1])
        pad = self.zeros((self.n_mels, max(0, padding)))
        spectogram = torch.cat((self.spectogram, spectogram, pad), -1)
        return spectogram if padding >= 0 else spectogram[-padding:]

    offset: int = 0

    async def pull(self) -> torch.Tensor:
        context = self.spectogram.shape[-1]
        iterator = self.window(self.buffer_nowait())
        async for frame in iterator:
            self.spectogram = torch.cat((self.spectogram, frame), -1)
        cutoff = min(context, max(self.spectogram.shape[-1] - N_FRAMES, 0))
        self.offset += cutoff
        self.spectogram = self.spectogram[:, cutoff:]
        return self.runoff()

    staging: Optional[Batcher] = None
    async def _push(self, sec: float, exact: bool = False) -> \
            AsyncIterator[torch.Tensor]:
        batching = int(sec * SAMPLE_RATE // HOP_LENGTH)
        init = self.window(self.buffer()) if self.staging is None \
                else self.staging
        self.staging = Batcher(init, batching, exact=exact)
        async for frame in self.staging:
            batched = batching if exact else frame.shape[-1]
            cutoff = max(self.spectogram.shape[-1] + batched - N_FRAMES, 0)
            self.offset += cutoff
            self.spectogram = torch.cat((
                    self.spectogram[:, cutoff:], frame), -1)
            yield self.runoff()

    reader: Optional[Awaitable] = None
    def start(self, **kw) -> None:
        if self.reader is None:
            self.reader = asyncio.create_task(self.read(**kw))

    async def push(self, sec: float, exact: bool=False, **kw) -> \
            AsyncIterator[torch.Tensor]:
        self.start(**kw)
        async for i in self._push(sec, exact):
            yield i
        assert self.reader is not None
        await self.reader

    async def request(self, sec: float, exact: bool=True, **kw) -> torch.Tensor:
        try:
            return await anext(self.push(sec, exact))
        except StopAsyncIteration:
            if self.reader is not None:
                await self.reader
            return self.zeros((self.n_mels, 0))

    async def full(self, **kw) -> torch.Tensor:
        await self.read(**kw)
        return await self.pull()

    def sequential(self, **kw) -> torch.Tensor:
        return asyncio.run(self.full(**kw))

    async def amplitudes(self, **kw) -> np.ndarray:
        self.start(**kw)
        res = []
        async for data in self.loader(self.buffer()):
            res.append(data)
        assert self.reader is not None
        await self.reader
        return np.concatenate(res)

    def all_amplitudes(self, **kw) -> np.ndarray:
        return asyncio.run(self.amplitudes(**kw))

class RawAudioFile(ArrayStream):
    def __init__(
            self, *, period: int = HOP_LENGTH, fname: PathType = 'out.raw',
            **kw):
        super().__init__(**kw)
        self.fname = fname
        self.period = period

    fp: Optional[IO[bytes]] = None
    async def read(self) -> None:
        fp = open(self.fname, 'rb') if self.fp is None else self.fp
        data = fp.read(self.period)
        while len(data) != 0:
            io_hold = self.write(data)
            assert io_hold is not None and self.write_blockable is True
            await io_hold
            data = fp.read(self.period)
        self.finished.set()

class AudioFile(RawAudioFile):
    def __init__(
            self, *, period: int = SAMPLE_RATE, fname: PathType = 'out.wav',
            **kw):
        assert not subprocess.run(
                ["which", "ffmpeg"], stdout=subprocess.PIPE).returncode
        super().__init__(period=period or -1, fname=fname, **kw)

    async def read(self) -> None:
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
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.fp = ps.stdout
        await super().read()
        _, stderr = ps.communicate()
        if ps.returncode not in (None, 0):
            raise RuntimeError(f"Failed to load audio: {stderr.decode()}")

class SequenceDone(Exception):
    pass

SeqType = Union[PathType, Tuple[PathType, ...]]

class AudioFileSequence(AudioFile):
    def __init__(self, *, seq: SeqType = '*.wav', **kw):
        global glob
        import glob
        super().__init__(fname="", **kw)
        _seq = (seq,) if isinstance(seq, (str, pathlib.Path)) else seq
        self.seq = tuple(j for i in _seq for j in glob.glob(str(i)))
        self.idx = 0

    def clear(self) -> None:
        self.finished.clear()

    def read(self) -> Coroutine[Any, Any, None]:
        async def inner():
            if self.idx >= len(self.seq):
                raise SequenceDone
            self.fname = self.seq[self.idx]
            await super(__class__, self).read()
            self.idx += 1
        self.clear()
        return inner()

class AudioFileStitch(ArrayStream):
    def __init__(
            self, *, seq: SeqType = "*.wav", period: int = SAMPLE_RATE, **kw):
        super().__init__(**kw)
        self.seq = AudioFileSequence(seq=seq, period=period, **kw)
        self.seq.q = self.q

    async def read(self) -> None:
        try:
            while True:
                await self.seq.read()
        except SequenceDone:
            self.finished.set()

class AudioCapture(ArrayStream):
    write_blockable: bool = False
    async def capture(self) -> None:
        raise NotImplementedError

linux: bool = False
try:
    # ALSA is a more typical setup than PortAudio on Linux
    import alsaaudio
    linux = True
except ImportError:
    import pyaudio

class ALSACapture(AudioCapture):
    def __init__(
            self, *, period: int = HOP_LENGTH, source: str = 'default',
            loop_delay: float = .001, **kw):
        super().__init__(**kw)
        self.period, self.source = period, source
        self.loop_delay = loop_delay

    async def busy_loop(self) -> None:
        inp = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK, channels=1,
                rate=self.rate, format=alsaaudio.PCM_FORMAT_S16_LE,
                periodsize=self.period, device=self.source)

        while True:
            l, data = inp.read()
            if l < 0:
                raise Exception(
                        "capture buffer overrun before write buffer")
            if l:
                self.write(data)
                await asyncio.sleep(self.loop_delay)

class PyAydioCapture(AudioCapture):
    def __init__(
            self, *, period: int = HOP_LENGTH, source: str = 'default',
            loop_delay: float = .001, **kw):
        super().__init__(**kw)
        self.period, self.loop_delay = period, loop_delay
        self.interface = pyaudio.PyAudio()

    async def capture(self) -> None:
        inp = self.interface.open(
                format=pyaudio.paInt16, channels=1, rate=self.rate,
                input=True)

        while True:
            # overflows when loading model, callback mode may be more reliable
            self.write(inp.read(self.period, exception_on_overflow=False))
            await asyncio.sleep(self.loop_delay)

# there are simpler ways, but none that the MyPy is happy about
class PlatformInheritance(type):
    def __new__(cls, clsname, bases, attrs):
        return super().__new__(cls, clsname, (
                ALSACapture if linux else PyAydioCapture,), attrs)

class LiveCapture(AudioCapture, metaclass=PlatformInheritance):
    async def read(self, length: Optional[float] = None) -> None:
        if length is None:
            await self.capture()
        else:
            try:
                await asyncio.wait_for(self.capture(), length)
            except TimeoutError:
                self.finished.set()

    def blocking(self, length: Optional[float] = None) -> None:
        asyncio.run(self.read(length))

class Recorder(LiveCapture):
    def __init__(self, *, fname: PathType = 'out.raw', **kw):
        super().__init__(**kw)
        self.fp = open(fname, 'wb')

    def write(self, data: bytes) -> None:
        super().write(data)
        self.fp.write(data)


import asyncio, os, sys
from whisper import load_model
from transcribe import Transcriber
from utils import PassthroughProperty
from audio import LiveCapture, AudioFileStitch, Recorder
from whisper.audio import CHUNK_LENGTH, FRAMES_PER_SECOND

def hms(sec):
    trim = sec < 3600
    h = "" if trim else str(int(sec) // 3600) + ":"
    m_fill = " " if trim else "0"
    m = "   " if sec < 60 else str(int(sec) // 60 % 60).rjust(2, m_fill) + ":"
    s = str(int(sec) % 60).rjust(2, '0') + "."
    c = str(round((sec % 1) * 100)).rjust(2, '0')
    return h + m + s + c

def tod(seconds):
    return time.strftime("%H:%M:%S", time.localtime(seconds))

class WatchJoin(metaclass=PassthroughProperty.defaults):
    def __init__(self, transform=repr, buffer=1_000):
        self.transform, self.buffer = transform, buffer

    skipped, scrollback = 0, 0
    @PassthroughProperty(()).setter
    def written(self, value):
        self._written = value[-self.buffer:]
        self.skipped = max(0, len(value) - self.buffer)

    @staticmethod
    def clear_line():
        if os.name == 'nt': # Windows
            print("\r" + chr(27) + "[2K", end="") #]
        else:
            print("\r" + chr(27) + "[0K", end="") #]

    @staticmethod
    def backtrack():
        print("\033[F", end="") #]

    def __call__(self, value):
        if len(value) < self.skipped:
            raise Exception()
        update = []
        for i in range(min(len(value) - self.skipped, len(self.written))):
            if value[self.skipped + i] != self.written[i]:
                update.append(self.skipped + i - len(value))
        adding = len(value) - self.skipped - len(self.written)
        self.scrollback = max(0, self.scrollback - adding)
        self.written = value
        for i in range(max(0, -adding)):
            self.backtrack()
            self.clear_line()
        init = pos = max(0, adding)
        for i in reversed(update):
            for j in range(pos, -i):
                self.backtrack()
            pos = -i
            self.clear_line()
            print(self.transform(value[i]), end="")
        print("\n" * max(0, pos - init), end="")
        for i in range(-init, 0):
            print(self.transform(value[i]))
        sys.stdout.flush()

class MinimalTranscriber(Transcriber):
    exact, chlen = True, CHUNK_LENGTH
    async def loop(self, stream, **kw):
        data = await stream.request(self.chlen, self.exact)
        while data.shape[-1] > 0:
            self(data, stream.offset, True)
            t = self.chlen - (stream.offset + data.shape[-1] - self.seek) \
                    / FRAMES_PER_SECOND + CHUNK_LENGTH
            data = await stream.request(t, self.exact)
        return self.all_segments

class AudioTranscriber(Transcriber):
    async def loop(self, stream, sec, **kw):
        async for data in stream.push(sec, **kw):
            self.restore(stream.offset)
            yield self(data, stream.offset, True)

    def gutter(self, segment):
        return str(segment["id"]).rjust(4) + "  " + hms(segment["start"])

    def repr(self, segment):
        return self.gutter(segment) + "    " + segment["text"]

    streamer = LiveCapture
    def stdout(self, sec=1, exact=False, **kw):
        kw["n_mels"] = self.model.dims.n_mels
        stream, printer = self.streamer(**kw), WatchJoin(self.repr)
        async def inner():
            print("Starting transcription...")
            async for out in self.loop(stream, sec, exact=exact):
                printer(out["segments"])
        asyncio.run(inner())

class RecorderTranscriber(AudioTranscriber):
    def __init__(self, *a, fname='out.json', **kw):
        global json
        import json
        self.fname = fname

    streamer = Recorder
    async def loop(self, *a, **kw):
        async for data in super().loop(*a, **kw):
            with open(self.fname, 'w') as fp:
                json.dump(data, fp)
            yield data

class ToDTranscriber(AudioTranscriber):
    def __init__(self, *a, **kw):
        global time
        import time
        self.initial = time.time()
        super().__init__(*a, **kw)

    def gutter(self, segment):
        return str(segment["id"]).rjust(4) + "  " + tod(
                self.initial + segment["start"])

class EnTranscriber(ToDTranscriber):
    _language = "en"

if __name__ == "__main__":
    EnTranscriber(load_model("base.en")).stdout(3)


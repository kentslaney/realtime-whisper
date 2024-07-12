import asyncio, os, sys
from transcribe import Transcriber
from utils import PassthroughProperty, PathType
from audio import LiveCapture, AudioFileStitch, Recorder, ArrayStream
from whisper.audio import CHUNK_LENGTH, FRAMES_PER_SECOND
from typing import Generic, TypeVar, Callable, List
from collections.abc import AsyncGenerator

def hms(sec: float) -> str:
    trim = sec < 3600
    h = "" if trim else str(int(sec) // 3600) + ":"
    m_fill = " " if trim else "0"
    m = "   " if sec < 60 else str(int(sec) // 60 % 60).rjust(2, m_fill) + ":"
    s = str(int(sec) % 60).rjust(2, '0') + "."
    c = str(round((sec % 1) * 100)).rjust(2, '0')
    return h + m + s + c

def tod(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(seconds))

T = TypeVar("T")

class WatchJoin(Generic[T], metaclass=PassthroughProperty.defaults):
    def __init__(
            self, transform: Callable[[T], str] = repr, buffer: int = 1_000):
        self.transform, self.buffer = transform, buffer

    skipped: int = 0
    scrollback: int = 0
    @PassthroughProperty(()).setter
    def written(self, value: List[T]):
        self._written = value[-self.buffer:]
        self.skipped = max(0, len(value) - self.buffer)

    @staticmethod
    def clear_line() -> None:
        if os.name == 'nt': # Windows
            print("\r" + chr(27) + "[2K", end="") #]
        else:
            print("\r" + chr(27) + "[0K", end="") #]

    @staticmethod
    def backtrack():
        print("\033[F", end="") #]

    def __call__(self, value: List[T]) -> None:
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
    exact: bool = True
    chlen: float = CHUNK_LENGTH
    async def loop(self, stream: ArrayStream, **kw) -> List[dict]:
        data = await stream.request(self.chlen, self.exact)
        while data.shape[-1] > 0:
            self(data, stream.offset, True)
            t = self.chlen - (stream.offset + data.shape[-1] - self.seek) \
                    / FRAMES_PER_SECOND + CHUNK_LENGTH
            data = await stream.request(t, self.exact)
        return self.all_segments

class AudioTranscriber(Transcriber):
    async def loop(self, stream: ArrayStream, sec: float, **kw) -> \
            AsyncGenerator[List[dict]]:
        async for data in stream.push(sec, **kw):
            self.restore(stream.offset)
            yield self(data, stream.offset, True)

    def gutter(self, segment: dict) -> str:
        return str(segment["id"]).rjust(4) + "  " + hms(segment["start"])

    def repr(self, segment: dict) -> str:
        return self.gutter(segment) + "    " + segment["text"]

    streamer: ArrayStream = LiveCapture
    def stdout(self, sec: float = 1., exact: bool = False, **kw) -> None:
        kw["n_mels"] = self.model.dims.n_mels
        stream, printer = self.streamer(**kw), WatchJoin(self.repr)
        async def inner():
            print("Starting transcription...")
            async for out in self.loop(stream, sec, exact=exact):
                printer(out["segments"])
        asyncio.run(inner())

class RecorderTranscriber(AudioTranscriber):
    def __init__(self, *a, fname: PathType = 'out.json', **kw):
        global json
        import json
        self.fname = fname

    streamer: ArrayStream = Recorder
    async def loop(self, *a, **kw) -> AsyncGenerator[List[dict]]:
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

    def gutter(self, segment: dict) -> str:
        return str(segment["id"]).rjust(4) + "  " + tod(
                self.initial + segment["start"])

class EnTranscriber(ToDTranscriber):
    _language = "en"

if __name__ == "__main__":
    from whisper import load_model
    EnTranscriber(load_model("base.en")).stdout(3)


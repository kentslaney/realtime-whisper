import asyncio, os, sys, time, json
from transcribe import Transcriber
from utils import (
        PassthroughProperty, PassthroughPropertyDefaults, PathType, ceildiv)
from audio import LiveCapture, AudioFileStitch, Recorder, ArrayStream, AudioFile
from whisper.audio import CHUNK_LENGTH, FRAMES_PER_SECOND
from typing import Generic, TypeVar, Callable, List, Union, Tuple, Optional
from collections.abc import AsyncIterator

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

class WatchJoin(Generic[T], metaclass=PassthroughPropertyDefaults):
    def __init__(
            self, transform: Callable[[T], str] = repr, buffer: int = 1_000):
        self.transform, self.buffer = transform, buffer

    skipped: int = 0
    scrollback: int = 0
    @PassthroughProperty[Union[List[T], Tuple]](()).setter
    def written(self, value: Union[List[T], Tuple]) -> None:
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
    async def process(self, stream: ArrayStream, **kw) -> dict:
        data = await stream.request(self.chlen, self.exact)
        while data.shape[-1] > 0:
            self(data, stream.offset, True)
            t = self.chlen - (stream.offset + data.shape[-1] - self.seek) \
                    / FRAMES_PER_SECOND + CHUNK_LENGTH
            data = await stream.request(t, self.exact)
        return self.result

class ProgressTranscriber(MinimalTranscriber):
    def __init__(self, *a, duration: Optional[float] = None, **kw):
        global tqdm
        import tqdm
        super().__init__(*a, **kw)
        self.duration, self.progress = duration, 0

    def __call__(self, *a, **kw) -> dict:
        if self._pbar is None:
            try:
                return super().__call__(*a, **kw)
            finally:
                self.close()
        else:
            return super().__call__(*a, **kw)

    @PassthroughProperty(None).property
    def pbar(self):
        if self._pbar is None:
            n = self.latest.shape[-1] if self.duration is None \
                    else ceildiv(self.duration * FRAMES_PER_SECOND, 1)
            self._pbar = tqdm.tqdm(
                    total=n, unit="frames", disable=self.verbose is not False)
            self._pbar.__enter__()
        return self._pbar

    def reporthook(self) -> None:
        update_to = min(self._seek, self.frame_offset + self.latest.shape[-1])
        self.pbar.update(update_to - self.progress)
        self.progress = update_to

    def close(self):
        self.pbar.__exit__(None, None, None)

    async def process(self, stream: ArrayStream, **kw) -> dict:
        self.pbar
        try:
            return await super().process(stream, **kw)
        finally:
            self.close()

    async def progressive(self, stream: AudioFile, **kw) -> dict:
        self.duration = stream.duration
        return await self.process(stream, **kw)

    def progressing(self, stream: AudioFile, **kw) -> dict:
        return asyncio.run(self.progressive(stream, **kw))

class AudioTranscriber(Transcriber):
    async def loop(self, stream: ArrayStream, sec: float, **kw) -> \
            AsyncIterator[dict]:
        async for data in stream.push(sec, **kw):
            self.restore(stream.offset)
            yield self(data, stream.offset, True)

    def gutter(self, segment: dict) -> str:
        return str(segment["id"]).rjust(4) + "  " + hms(segment["start"])

    def repr(self, segment: dict) -> str:
        return self.gutter(segment) + "    " + segment["text"]

    streamer: type[ArrayStream] = LiveCapture
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
        self.fname = fname

    streamer: type[ArrayStream] = Recorder
    async def loop(self, *a, **kw) -> AsyncIterator[dict]:
        async for data in super().loop(*a, **kw):
            with open(self.fname, 'w') as fp:
                json.dump(data, fp)
            yield data

class ToDTranscriber(AudioTranscriber):
    def __init__(self, *a, **kw):
        self.initial = time.time()
        super().__init__(*a, **kw)

    def gutter(self, segment: dict) -> str:
        return str(segment["id"]).rjust(4) + "  " + tod(
                self.initial + segment["start"])

class EnTranscriber(ToDTranscriber):
    _language = "en"

if __name__ == "__main__":
    from whisper import load_model
    EnTranscriber(load_model("large")).stdout(5)


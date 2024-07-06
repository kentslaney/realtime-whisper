import asyncio, os, sys
from whisper import load_model
from transcribe import Transcriber
from utils import PassthroughProperty
from audio import LiveCapture, AudioFileStitch
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
    async def loop(self, stream, **kw):
        breakpoint()
        for i in range(20):
            data = await stream.request(18)
            print(data, data.shape, stream.offset)
        raise Exception # TODO
        exact = True
        chlen = 25
        data = await stream.request(chlen, exact)
        while len(data) > 0:
            self(data, stream.offset)
            t = chlen - CHUNK_LENGTH - (
                    stream.offset + len(data) - self.seek) / FRAMES_PER_SECOND
            print(t)
            data = await stream.request(t, exact)
            print(data)
        return self.all_segments

class AudioTranscriber(Transcriber):
    async def loop(self, stream, sec, **kw):
        async for data in stream.push(sec, **kw):
            self.restore(stream.offset)
            yield self(data, stream.offset)

    def gutter(self, segment):
        return str(segment["id"]).rjust(4) + "  " + hms(segment["start"])

    def repr(self, segment):
        return self.gutter(segment) + "    " + segment["text"]

    def stdout(self, sec=1, exact=False, **kw):
        stream, printer = LiveCapture(**kw), WatchJoin(self.repr)
        async def inner():
            print("Starting transcription...")
            async for out in self.loop(stream, sec, exact=exact):
                printer(out["segments"])
        asyncio.run(inner())

class ToDTranscriber(AudioTranscriber):
    def __init__(self, *a, **kw):
        global time
        import time
        self.initial = time.time()
        super().__init__(*a, **kw)

    def gutter(self, segment):
        return str(segment["id"]).rjust(4) + "  " + tod(
                self.initial + segment["start"])

class EnTranscriber(AudioTranscriber):
    _language = "en"

if __name__ == "__main__":
    # ToDTranscriber(load_model("large")).stdout(5, n_mels=128)
    EnTranscriber(load_model("tiny.en")).stdout(3)

import asyncio, os, sys
from whisper import load_model
from transcribe import Transcriber
from audio import LiveCapture, AudioFileStitch

def print_inline(value, end=""):
    if os.name == 'nt': # Windows
        print("\r" + chr(27) + "[2K", end="") #]
    else:
        print("\r" + chr(27) + "[0K", end="") #]
    print(value, end=end)
    sys.stdout.flush()

def hms(sec):
    trim = sec < 3600
    h = "" if trim else str(int(sec) // 3600) + ":"
    m_fill = " " if trim else "0"
    m = "   " if sec < 60 else str(int(sec) // 60 % 60).rjust(2, m_fill) + ":"
    s = str(int(sec) % 60).rjust(2, '0') + "."
    c = str(round((sec % 1) * 100)).rjust(2, '0')
    return h + m + s + c

class EnTranscriber(Transcriber):
    _language = "en"

class AudioTranscriber(EnTranscriber):
    async def loop(self, stream, sec, **kw):
        async for data in stream.push(sec, **kw):
            yield self(data, stream.offset)
            self.restore()

    def gutter(self, segment):
        # return hms(segment["start"]) + " - " + hms(segment["end"]) + "   "
        return str(segment["id"]).rjust(4) + "  " + \
                hms(segment["start"]) + "   "

    def stdout(self, sec=1, exact=False, **kw):
        stream = LiveCapture(**kw)
        async def inner(): # TODO: make stdout closer to final transcription
            pending = 0
            print("Starting transcription...")
            async for out in self.loop(stream, sec, exact=exact):
                for line in out["segments"][pending:-1]:
                    print_inline(self.gutter(line) + line["text"], "\n")
                if len(out["segments"]) == 0:
                    continue
                current = out["segments"][-1]
                print_inline(self.gutter(current) + current["text"])
                pending = len(out["segments"]) - 1
        asyncio.run(inner())

def tod(seconds):
    return time.strftime("%H:%M:%S", time.localtime(seconds))

class ToDTranscriber(AudioTranscriber):
    def __init__(self, *a, **kw):
        global time
        import time
        self.initial = time.time()
        super().__init__(*a, **kw)

    def gutter(self, segment):
        return str(segment["id"]).rjust(4) + "  " + \
                tod(self.initial + segment["start"]) + "   "

if __name__ == "__main__":
    # ToDTranscriber(load_model("large")).stdout(5, n_mels=128)
    ToDTranscriber(load_model("medium")).stdout(5)

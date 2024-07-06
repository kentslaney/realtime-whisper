import torch

import sys, os.path; end_locals, start_locals = lambda: sys.path.pop(0), (
    lambda x: x() or x)(lambda: sys.path.insert(0, os.path.dirname(__file__)))

from audio import *
from transcribe import Transcriber
from interface import *
from utils import Batcher

end_locals()

async def test_range(x, y):
    for i in range(x):
        yield np.arange(1000 * i, 1000 * i + y)

def test_resample(x, y, z):
    res = []
    async def inner():
        async for i in Batcher(test_range(x, y), z, exact=True):
            res.append(i)
    asyncio.run(inner())
    return res

# if __name__ == "__main__":
#     print(test_resample(6, 7, 3))

class LenTest(AudioFile):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.total = 0

    def write(self, data):
        super().write(data)
        self.total += self.q.get_nowait().size

    async def test(self):
        await self.read()
        print(self.total)

    def __call__(self):
        asyncio.run(self.test())

class Test(AudioFileStitch):
    def __init__(self, **kw):
        global plt
        import matplotlib.pyplot as plt
        super().__init__(**kw)

    def padding(self, content_frames):
        return N_FRAMES - content_frames

    async def test(self):
        out = await self.full()
        plt.imshow(out)
        plt.show()

    def __call__(self):
        asyncio.run(self.test())

class MockTokenizer:
    def __init__(self, language, **kw):
        self.language = language
        for k, v in kw.items():
            setattr(self, k, v)

    def encode(prompt):
        return [self.language, self, prompt]

class TranscriberTest(Transcriber):
    dtype = torch.float32
    model = type("MockModel", (), {
            "is_multilingual": True,
            "num_languages": None,
            "device": torch.device("cpu")
        })()
    latest = None
    _seek = 0.

    def __init__(self, seq):
        super().__init__(self.model)
        self.seq = seq
        self.result = []
        self.result.append(self.initial_prompt_tokens)
        self.latest = []
        for i in range(len(self.seq)):
            self._seek = i
            self.result.append(self.initial_prompt_tokens)

    def detect_language(self):
        self.result.append("sample")
        return self.seq[self._seek]

    def get_tokenizer(self, language, **kw):
        return MockTokenizer(language, **kw)

class ReadableMinimal(MinimalTranscriber, AudioTranscriber):
    def gutter(self, segment):
        return hms(segment["start"]) + " - " + hms(segment["end"])

    def __repr__(self):
        return "\n".join(map(self.repr, self.all_segments))

def MinimalTest():
    model = ReadableMinimal(load_model("base.en"))
    asyncio.run(model.loop(AudioFileStitch()))
    return model

if __name__ == "__main__":
    print(MinimalTest())


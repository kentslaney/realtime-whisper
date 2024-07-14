import torch, pathlib

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
        # global plt
        # import matplotlib.pyplot as plt
        super().__init__(**kw)

    def padding(self, content_frames):
        return N_FRAMES - content_frames

    async def test(self):
        out = await self.full()
        # plt.imshow(out)
        # plt.show()

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
    _seek = 0

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

def match2d(a, b, eps=1e-6):
    assert b.shape[-1] < a.shape[-1]
    c = np.concatenate((a, np.zeros(a.shape[:-1] + (b.shape[-1],))), -1)
    d = b.numpy()
    res = np.where(
            np.vectorize(lambda i: np.all(np.isclose(
                d, c[:, i : b.shape[-1] + i], 0, eps)))(np.arange(a.shape[-1])))
    if res[0].shape:
        return res
    err = np.min(np.vectorize(
            lambda i: np.max(np.abs(a[:, i : b.shape[-1] + i] - b)))(
                np.arange(a.shape[-1] - b.shape[-1])))
    breakpoint()

class ModelContainer:
    def __init__(self, model, idx=None, ref=0.):
        self._model, self._idx, self._ref = model, idx, ref
        self._decoded, self._options, self._results = [], [], []

    def decode(self, segment, options):
        self._options.append(options)
        if self._ref is None or options.temperature == self._ref:
            if isinstance(self._idx, Transcriber):
                self._decoded.append(self._idx.seek)
            elif self._idx is not None:
                self._decoded.append(match2d(self._idx, segment))
            else:
                self._decoded.append(segment)
        res = self._model.decode(segment, options)
        self._results.append(res)
        return res

    def __getattr__(self, key):
        return getattr(self._model, key)

test_dir = pathlib.Path(__file__).parents[0] / "tests"
test_files = str(test_dir / "*.wav")
test_file = str(test_dir / "List01Sentence01.wav")

def minimal_test(seq=test_files):
    from whisper import load_model, transcribe
    model = load_model("base.en")
    stream = lambda: AudioFileStitch(seq=seq)
    mel = stream().sequential()
    def transcriber(idx=...):
        container = ModelContainer(
                model, *(() if idx in (None, ...) else (idx,)), ref=None)
        res = ReadableMinimal(container)
        if idx is ...:
            container._idx = res
        return res

    minimal = transcriber()
    asyncio.run(minimal.process(stream()))

    polyfill = transcriber()
    polyfill(mel)

    from whisper.audio import log_mel_spectrogram
    amps = stream().all_amplitudes()
    mel_original = log_mel_spectrogram(amps)
    # original = transcriber(mel_original)
    original = transcriber()
    original.all_segments = transcribe(original.model, amps)['segments']

    return minimal, polyfill, original

def mel_test(seq=test_files, check_amp=False):
    from whisper.audio import log_mel_spectrogram
    stream = lambda: AudioFileStitch(seq=seq)
    original = log_mel_spectrogram(stream().all_amplitudes())
    polyfill = stream().sequential()[:, :-N_FRAMES]
    if isinstance(seq, str) and "*" not in seq and check_amp:
        assert torch.all(original == log_mel_spectrogram(seq))
    return polyfill, original

if __name__ == "__main__":
    torch.set_printoptions(edgeitems=8, linewidth=200)
    np.set_printoptions(edgeitems=8, linewidth=200)
    print(*minimal_test(), sep="\n\n")


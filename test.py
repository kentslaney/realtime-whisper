import sys, os.path; end_locals, start_locals = lambda: sys.path.pop(0), (
    lambda x: x() or x)(lambda: sys.path.insert(0, os.path.dirname(__file__)))

from audio import *

end_locals()

async def test_range(x, y):
    for i in range(x):
        yield np.arange(1000 * i, 1000 * i + y)

def test_resample(x, y, z):
    res = []
    async def inner():
        async for i in resample(test_range(x, y), z):
            res.append(i)
    asyncio.run(inner())
    return res

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


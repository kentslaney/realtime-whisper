import collections, torch
import numpy as np

# boilerplate for property with _{name} storage and passthrough getter/setter
class PassthroughProperty:
    def __init__(self, default):
        self.value = default

    f = None
    def setter(self, f):
        self.f = f
        return self

    g = None
    def property(self, g):
        self.g = property(g)
        return self

    @staticmethod
    def defaults(clsname, bases, attrs):
        def closure(f, v):
            def prop(self):
                return getattr(self, v)
            def setter(self, value):
                setattr(self, v, value)
            prop.__name__ = setter.__name__ = f
            return property(prop), setter

        updates = {}
        for k, v in attrs.items():
            if not isinstance(v, PassthroughProperty):
                continue
            private = "_" + k
            assert private not in attrs
            updates[private] = v.value
            getter, setter = closure(k, private)
            updates[k] = (v.g or getter).setter(v.f or setter)
        return type(clsname, bases, {**attrs, **updates})

class Unwrap:
    def __init__(self, iterator):
        while isinstance(iterator, PassthroughTransform):
            iterator = iterator.handoff()
        if isinstance(iterator, __class__):
            self._initial, self.started = iterator.initial(), iterator.started
            if iterator.started:
                iterator = iterator.iterator
            else:
                self.iterator = iterator.iterator
                return
        elif not isinstance(iterator, collections.abc.AsyncIterator):
            iterator = aiter(iterator)
        try:
            self._initial = anext(iterator)
            self.iterator, self.started = iterator, False
        except StopAsyncIteration:
            self.iterator, self.started = iter(()), True

    async def initial(self):
        while isinstance(self._initial, collections.abc.Awaitable):
            self._initial = await self._initial
        return self._initial

    async def iter(self):
        if not self.started:
            self.started = True
            yield await self.initial()
        async for i in self.iterator:
            yield i

    def __aiter__(self):
        return self.iter()

    async def prop(self, key, default):
        if hasattr(self, "initial"):
            return getattr(await self.initial(), key)
        else:
            return default

    @property
    def shape(self):
        return self.prop("shape", ())

    @property
    def dtype(self):
        return self.prop("dtype", None)

    @property
    async def concat(self):
        return np.concatenate if isinstance(await self.dtype, np.dtype) \
                else torch.cat

class PassthroughTransform:
    def handoff(self):
        raise NotImplementedError

class BoxedIterator(PassthroughTransform):
    def __init__(self, iterator):
        self.iterator = iterator
        self.flag = object()

    def handoff(self):
        self.flag = None
        return self.iterator

    def __aiter__(self):
        return self.iter()

    async def iter(self):
        if self.flag is None:
            raise Exception("iterator source removed")
        self.flag = flag = object()
        async for i in self.iterator:
            yield i
            if self.flag != flag:
                raise Exception("source can only be used by one iterator")

def LookAlong(axis):
    assert axis >= 0
    empties = (slice(None),) * axis

    class LookAlong:
        def __init__(self, value):
            self.value = value

        @property
        def shape(self):
            return self.value.shape[axis]

        def __getitem__(self, idx):
            return self.value[empties + (idx,)]

    return LookAlong

class PassthroughMap(PassthroughTransform):
    def __init__(self, apply, iterator):
        self.iterator, self.apply = iterator, apply

    def handoff(self):
        return self.iterator

    async def iter(self):
        async for i in self.iterator:
            yield self.apply(i)

    def __aiter__(self):
        return self.iter()

class Group:
    def __init__(self, concat):
        self.concat = concat
        self.holding = []
        self.consumed = 0
        self.shape = 0

    def add(self, value):
        self.holding.append(value)
        self.shape += value.shape

    def take(self, amount, exact=True):
        assert amount > 0 and amount <= self.shape
        self.shape -= amount
        taking, start = -self.consumed, self.consumed
        for i, x in enumerate(self.holding):
            taking += x.shape
            if taking >= amount:
                self.consumed = amount - taking + x.shape
                break
        if taking == amount or not exact:
            self.shape += amount - taking
            self.consumed = 0
            res = self.concat([self.holding[0][start:]] + self.holding[1 : i])
            self.holding = self.holding[i + 1:]
            return res
        if i == 0:
            return self.holding[0][start:self.consumed]
        res = self.concat(
                [self.holding[0][start:]] + self.holding[1 : i - 1] +
                [self.holding[i][:self.consumed]])
        self.holding = self.holding[i:]
        return res

class Taken:
    def take(self, *a, **kw):
        raise Exception("batch queue moved")

class Batcher(PassthroughTransform):
    def __init__(self, iterator, size, axis=-1, exact=False):
        assert isinstance(size, int) and size > 0
        self.size, self._axis, self.exact = size, axis, exact
        if isinstance(iterator, __class__):
            self.group = iterator.group
        self.preview = Unwrap(iterator)

    _iterator = None
    @property
    async def iterator(self):
        if self._iterator is None:
            self.axis = len(await self.preview.shape) + self._axis \
                    if self._axis < 0 else self._axis
            if not hasattr(self, "group"):
                self.group = Group(await self.preview.concat)
            self._iterator = PassthroughMap(
                    LookAlong(self.axis), BoxedIterator(self.preview))
        return self._iterator

    def handoff(self):
        self.group = Taken()
        return self.preview if self._iterator is None else self._iterator

    def __aiter__(self):
        return self

    async def __anext__(self):
        iterator = aiter(await self.iterator)
        while self.group.shape < self.size:
            self.group.add(await anext(iterator))
        return self.group.take(self.size, self.exact)


import collections

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
            self.initial = iterator.initial
            if iterator.started:
                iterator = iterator.iterator
            else:
                self.iterator = iterator.iterator
                self.started = False
                return
        elif not isinstance(iterator, collections.abc.AsyncIterator):
            iterator = aiter(iterator)
        try:
            self.initial = await anext(iterator)
            self.iterator, self.started = iterator, False
        except StopAsyncIteration:
            self.iterator, self.started = iter(()), True

    def __aiter__(self):
        self.started = True
        yield self.initial
        async for i in self.iterator:
            yield i

    def prop(self, key, default):
        if hasattr(self, "initial"):
            return getattr(self.initial, key)
        else:
            return default

    @property
    def shape(self):
        return self.prop("shape", ())

    @property
    def dtype(self):
        return self.prop("dtype", None)

    @property
    def concat(self):
        return np.concatenate if isinstance(dtype, np.dtype) else torch.cat

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

    def __aiter__(self):
        async for i in self.iterator:
            yield self.apply(i)

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

class Batcher:
    def __init__(self, iterator, size, running=None, axis=-1, exact=False):
        assert isinstance(size, int)
        self.iterator, self.size, self.running, self.axis, self.exact = \
                iterator, size, running, axis, exact
        if isinstance(running, __class__):
            pass

    def __aiter__(self):
        pass

def tmp(axis, iterator):
    preview = Unwrap(iterator)
    iterator = PassthroughMap(LookAlong(axis), BoxedIterator(preview))
    pass


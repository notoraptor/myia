"""Tools to intern the instances of certain classes."""

import weakref


_intern_pool = weakref.WeakValueDictionary()


pyhash = hash


class EqKey:
    """Base class for Atom/Elements."""

    def __init__(self, obj):
        """Initialize an EqKey."""
        t = type(obj)
        if t in (int, bool):
            t = float
        self.type = t


class Atom(EqKey):
    """Object with a single value to test for equality directly."""

    def __init__(self, obj, value):
        """Initialize an Atom."""
        super().__init__(obj)
        self.value = value


class Elements(EqKey):
    """Object with multiple values to process for equality recursively."""

    def __init__(self, obj, *values, values_iterable=None):
        """Initialize an Elements."""
        super().__init__(obj)
        if values_iterable is None:
            self.values = values
        else:
            assert not values
            self.values = values_iterable


def eqkey(x):
    """Return the equality key for x."""
    if getattr(x, '_incomplete', False):
        raise IncompleteException()
    elif isinstance(x, EqKey):
        return x
    elif isinstance(x, (list, tuple)):
        return Elements(x, *x)
    elif isinstance(x, (set, frozenset)):
        return Elements(x, values_iterable=frozenset(x))
    elif isinstance(x, dict):
        return Elements(x, *x.items())
    elif hasattr(x, '__eqkey__'):
        return x.__eqkey__()
    else:
        return Atom(x, x)


class RecursionException(Exception):
    """Raised when a data structure is found to be recursive."""


class IncompleteException(Exception):
    """Raised when a data structure is incomplete."""


def deep_eqkey(obj, path=frozenset()):
    """Return a key for equality tests for non-recursive structures."""
    cachable = getattr(obj, '__cache_eqkey__', False)
    if cachable:
        cached = getattr(obj, '_eqkey_deepkey', None)
        if cached is not None:
            return cached

    oid = id(obj)
    if oid in path:
        raise RecursionException()

    key = eqkey(obj)
    if isinstance(key, Elements):
        dk = key.type, type(key.values)(
            deep_eqkey(x, path | {oid}) for x in key.values
        )
    else:
        assert isinstance(key, Atom)
        dk = key.type, key.value

    if cachable:
        obj._eqkey_deepkey = dk
    return dk


def hashrec(obj, path, cache):
    """Hash a (possibly self-referential) object."""
    oid = id(obj)
    if oid in path:
        return 0
    if oid in cache:
        return cache[oid][1]
    path = path | {oid}

    key = eqkey(obj)

    if isinstance(key, Atom):
        rval = pyhash((key.type, key.value))

    elif isinstance(key, Elements):
        subhash = [hashrec(x, path, cache) for x in key.values]
        rval = pyhash((key.type, type(key.values)(subhash)))

    else:
        raise AssertionError()

    cache[oid] = obj, rval
    return rval


def eqrec(obj1, obj2, path1=frozenset(), path2=frozenset(), cache=None):
    """Compare two (possibly self-referential) objects for equality."""
    id1 = id(obj1)
    id2 = id(obj2)

    if (id1, id2) in cache:
        return True

    if id1 in path1 or id2 in path2:
        return False

    if obj1 is obj2:
        return True

    path1 = path1 | {id1}
    path2 = path2 | {id2}
    cache.add((id1, id2))

    key1 = eqkey(obj1)
    key2 = eqkey(obj2)

    if type(key1) is not type(key2) or key1.type is not key2.type:
        return False

    if isinstance(key1, Atom):
        return key1.value == key2.value

    elif isinstance(key1, Elements):
        v1 = key1.values
        v2 = key2.values
        if len(v1) != len(v2):
            return False
        if isinstance(v1, frozenset):
            # TODO: save on complexity by sorting v1/v2 by hash?
            v2 = list(v2)
            for x1 in v1:
                for i, x2 in enumerate(v2):
                    if eqrec(x1, x2, path1, path2, cache):
                        break
                else:
                    return False
                del v2[i]
            else:
                assert not v2
                return True
        else:
            for x1, x2 in zip(v1, v2):
                if not eqrec(x1, x2, path1, path2, cache):
                    return False
            else:
                return True

    else:
        raise AssertionError()


def hash(obj):
    """Hash a (possibly self-referential) object."""
    try:
        return pyhash(deep_eqkey(obj))

    except RecursionException:
        return hashrec(obj, frozenset(), {})


def eq(obj1, obj2):
    """Compare two (possibly self-referential) objects for equality."""
    try:
        key1 = deep_eqkey(obj1)
        key2 = deep_eqkey(obj2)
        return key1 == key2

    except RecursionException:
        return eqrec(obj1, obj2, frozenset(), frozenset(), set())


class Wrapper:
    """Wraps an object and uses eq/hash for equality."""

    def __init__(self, obj):
        """Initialize a Wrapper."""
        self._obj = weakref.ref(obj)

    def __eq__(self, other):
        return eq(self._obj(), other._obj())

    def __hash__(self):
        return hash(self._obj())


class InternedMC(type):
    """Metaclass for a class where all members are interned."""

    def new(cls, *args, **kwargs):
        """Instantiates a non-interned instance."""
        obj = object.__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

    def intern(cls, inst):
        """Get the interned instance."""
        wrap = Wrapper(inst)
        try:
            existing = _intern_pool.get(wrap, None)
        except IncompleteException:
            return inst
        if existing is None:
            _intern_pool[wrap] = inst
            try:
                inst._canonical = True
            except Exception:
                pass
            return inst
        else:
            return existing

    def __call__(cls, *args, **kwargs):
        """Instantiates an interned instance."""
        inst = cls.new(*args, **kwargs)
        return inst.intern()


class Interned(metaclass=InternedMC):
    """Instances of this class are interned.

    Using the __eqkey__ method to generate a key for equality purposes, each
    instance with the same eqkey is mapped to a single canonical instance.
    """

    def intern(self):
        """Get the interned version of the instance."""
        return InternedMC.intern(type(self), self)

    def __eqkey__(self):
        """Generate a key for equality/hashing purposes."""
        raise NotImplementedError('Implement in subclass')


class PossiblyRecursive:
    """Base class for data that might be recursive."""

    @classmethod
    def empty(cls):
        """Create an empty, incomplete instance."""
        inst = object.__new__(cls)
        inst._incomplete = True
        return inst

    def __init__(self):
        """Initialization sets the object to complete."""
        self._incomplete = False
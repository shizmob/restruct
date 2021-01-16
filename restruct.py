import os
from io import BytesIO
import errno
import math
import types
import struct
import enum
import collections
import itertools
from contextlib import contextmanager

from typing import Generic as G, Union as U, TypeVar, Any, Callable, Sequence, Mapping, Optional as O, Tuple as TU


## Helpers

def indent(s: str, count: int, start: bool = False) -> str:
    """ Indent all lines of a string. """
    lines = s.splitlines()
    for i in range(0 if start else 1, len(lines)):
        lines[i] = ' ' * count + lines[i]
    return '\n'.join(lines)

def stretch(b: bytes, count: int) -> bytes:
    b *= count // len(b)
    b += b[:count - len(b)]
    return b

def format_value(value: Any, formatter: Callable[[Any], str], indentation: int = 0) -> str:
    """ Format containers to use the given formatter function instead of always repr(). """
    if isinstance(value, (dict, collections.Mapping)):
        if value:
            fmt = '{{\n{}\n}}'
            values = [indent(',\n'.join('{}: {}'.format(
                format_value(k, formatter),
                format_value(v, formatter)
            ) for k, v in value.items()), 2, True)]
        else:
            fmt = '{{}}'
            values = []
    elif isinstance(value, (list, set, frozenset)):
        l = len(value)
        is_set = isinstance(value, (set, frozenset))
        if l > 3:
            fmt = '{{\n{}\n}}' if is_set else '[\n{}\n]'
            values = [indent(',\n'.join(format_value(v, formatter) for v in value), 2, True)]
        elif l > 0:
            fmt = '{{{}}}' if is_set else '[{}]'
            values = [', '.join(format_value(v, formatter) for v in value)]
        else:
            fmt = '{{}}' if is_set else '[]'
            values = []
    elif isinstance(value, (bytes, bytearray)):
        fmt = '{}'
        values = [format_bytes(value)]
    else:
        fmt = '{}'
        values = [formatter(value)]
    return indent(fmt.format(*values), indentation)

def format_bytes(bs: bytes) -> str:
    return '[' + ' '.join(hex(b)[2:].zfill(2) for b in bs) + ']'

def format_path(path: Sequence[str]) -> str:
    s = ''
    first = True
    for p in path:
        sep = '.'
        if isinstance(p, int):
            p = '[' + str(p) + ']'
            sep = ''
        if sep and not first:
            s += sep
        s += p
        first = False
    return s

def class_name(s: Any, module_whitelist: Sequence[str] = {'__main__', 'builtins', __name__}) -> str:
    if not isinstance(s, type):
        s = s.__class__
    module = s.__module__
    name = s.__qualname__
    if module in module_whitelist:
        return name
    return module + '.' + name

def friendly_name(s: Any) -> str:
    if hasattr(s, '__name__'):
        return s.__name__
    return str(s)

def process_sizes(s: Mapping[str, int], cb: Callable[[int, int], int]) -> Mapping[str, int]:
    sizes = {}
    for prev in s:
        for k, n in prev.items():
            p = sizes.get(k, 0)
            if p is None or n is None:
                sizes[k] = None
            else:
                sizes[k] = cb(p, n)
    return sizes

def min_sizes(*s: Mapping[str, int]) -> Mapping[str, int]:
    return process_sizes(s, min)

def max_sizes(*s: Mapping[str, int]) -> Mapping[str, int]:
    return process_sizes(s, max)

def add_sizes(*s: Mapping[str, int]) -> Mapping[str, int]:
    return process_sizes(s, lambda a, b: a + b)

def ceil_sizes(s: Mapping[str, U[int, float]]) -> Mapping[str, int]:
    d = {}
    for k, v in s.copy().items():
        d[k] = math.ceil(v)
    return d


@contextmanager
def seeking(fd: 'IO', pos: int, whence: int = os.SEEK_SET) -> None:
    oldpos = fd.tell()
    fd.seek(pos, whence)
    try:
        yield fd
    finally:
        fd.seek(oldpos, os.SEEK_SET)


## Bases 

class BitAlignment(enum.Enum):
    No = enum.auto()
    Fill = enum.auto()
    Yes = enum.auto()

class IO:
    __slots__ = ('handle', 'bit_left', 'bit_val', 'bit_align')

    def __init__(self, handle, bit_align = BitAlignment.No) -> None:
        self.handle = handle
        self.bit_left = 0
        self.bit_val = None
        self.bit_align = bit_align

    def get_bits(self, n: int) -> TU[int, int]:
        if n > 0 and self.bit_left == 0:
            self.bit_val = self.handle.read(1)[0]
            self.bit_left = 8
        nb = min(n, self.bit_left)
        val = self.bit_val & ((1 << nb) - 1)
        self.bit_val >>= nb
        self.bit_left -= nb
        return n - nb, val

    def put_bits(self, val: int, n: int) -> TU[int, int]:
        if n > 0 and self.bit_left == 0:
            self.bit_left = 8
            self.bit_val = 0
        nb = min(n, self.bit_left)
        self.bit_val |= (val & ((1 << nb) - 1)) << (8 - self.bit_left)
        self.bit_left -= nb
        if nb > 0 and self.bit_left == 0:
            self.handle.write(bytes([self.bit_val]))
            self.bit_val = None
        return n - nb, (val >> nb)

    def flush_bits(self):
        if self.bits_left == 0:
            return
        self.put_bits(0, 8 - self.bit_left)

    def read(self, n: int = -1, *, bits: bool = False) -> U[bytes, int]:
        if bits:
            nl, v = self.get_bits(n)
            val = v << nl
            if nl >= 8:
                rounds = nl // 8
                nl -= 8 * rounds
                val |= int.from_bytes(self.handle.read(rounds), byteorder='big') << nl
            if nl > 0:
                _, v = self.get_bits(nl)
                val |= v
            return val
        if self.bit_left > 0:
            if self.bit_align == BitAlignment.No:
                raise ValueError('misaligned read')
            elif self.bit_align == BitAlignment.Yes:
                return self.read(n * 8, bits=True)
            elif self.bit_align == BitAlignment.Fill:
                self.bit_left = 0
                self.bit_val = None
        return self.handle.read(n)

    def write(self, b: U[bytes, int], *, bits: O[int] = None) -> None:
        if bits is not None:
            n, b = self.put_bits(b, bits)
            if n > 8:
                rounds = n // 8
                shift = 8 * rounds
                self.handle.write((b & ((1 << shift) - 1)).to_bytes(rounds, byteorder='big'))
                b >>= shift
                n -= shift
            if n > 0:
                self.put_bits(b, n)
            return
        if self.bit_left > 0:
            if self.bit_align == BitAlignment.No:
                raise ValueError('misaligned write')
            elif self.bit_align == BitAlignment.Yes:
                return self.write(int.from_bytes(b, byteorder='big'), bits=len(b) * 8)
            elif self.bit_align == BitAlignment.Fill:
                self.flush_bits()
        return self.handle.write(b)

    def flush(self):
        if self.bit_align == BitAlignment.Fill:
            self.flush_bits()
        return self.handle.flush()

    def seek(self, n: int, whence: int = os.SEEK_SET) -> None:
        return self.handle.seek(n, whence)

    def tell(self) -> int:
        return self.handle.tell()

    @property
    def root(self):
        handle = self.handle
        while hasattr(handle, 'root'):
            handle = handle.root
        return handle

    @contextmanager
    def wrapped(self, handle):
        self.flush()
        old = self.handle
        self.handle = handle
        yield self
        self.handle = old

class Stream:
    __slots__ = ('name', 'offset', 'dependents', 'pos')

    def __init__(self, name: str, dependents: Sequence['Stream'] = None) -> None:
        self.name = name
        self.offset = None
        self.dependents = dependents or []
        self.pos = None

    def reset(self):
        self.offset = self.pos = None

class Params:
    __slots__ = ('streams', 'default_stream', 'user')

    def __init__(self, streams: Sequence[Stream] = None):
        default = streams[0] if streams else Stream('default')
        self.streams = {s.name: s for s in (streams or [default, Stream('refs', [default])])}
        self.default_stream = default
        self.user = types.SimpleNamespace()

    def reset(self):
        for s in self.streams.values():
            s.reset()

class Context:
    __slots__ = ('root', 'value', 'path', 'stream_path', 'user', 'params')

    def __init__(self, root: 'Type', value: O[Any] = None, params: O[Params] = None) -> None:
        self.root = root
        self.value = value
        self.path = []
        self.stream_path = []
        self.params = params or Params()
        self.params.reset()
        self.user = self.params.user

    @contextmanager
    def enter(self, name: str, type: 'Type') -> None:
        self.path.append((name, type))
        yield
        self.path.pop()

    @contextmanager
    def enter_stream(self, stream: str, io: O[IO] = None, pos: O[int] = None, reference = os.SEEK_SET) -> None:
        stream = self.params.streams[stream]
        if io:
            if pos is None:
                if stream.offset is None:
                    stream.offset = self.stream_offset(stream)
                    stream.pos = stream.offset
                pos = stream.pos
            with seeking(io.root, pos, reference) as s, io.wrapped(s) as f:
                self.stream_path.append(stream)
                yield f
                self.stream_path.pop()
                stream.pos = f.tell()
        else:
            self.stream_path.append(stream)
            yield io
            self.stream_path.pop()

    def stream_offset(self, stream: Stream) -> int:
        size = 0
        for s in stream.dependents:
            size += self.stream_offset(s) + self.stream_size(s)
        return size

    def stream_size(self, stream: Stream) -> O[int]:
        return sizeof(self.root, self.value, self, stream=stream.name)

    def format_path(self) -> str:
        return format_path(name for name, parser in self.path)

class Error(Exception):
    __slots__ = ('path', 'stream_path')

    def __init__(self, context: Context, exception: Exception) -> None:
        path = context.format_path()
        if not isinstance(exception, Exception):
            exception = ValueError(exception)

        super().__init__('{}{}: {}'.format(
            ('[' + path + '] ') if path else '', class_name(exception), str(exception)
        ))
        self.exception = exception
        self.path = context.path.copy()
        self.stream_path = context.stream_path.copy()

class Type:
    __slots__ = ()

    def parse(self, io: IO, context: Context) -> Any:
        raise NotImplemented

    def emit(self, value: Any, io: IO, context: Context) -> None:
        raise NotImplemented

    def sizeof(self, value: O[Any], context: Context) -> O[int]:
        return None

    def default(self, context: Context) -> O[Any]:
        return None


## Type helpers

T = TypeVar('T', bound=Type)
T2 = TypeVar('T2', bound=Type)


## Base types

class Nothing(Type):
    __slots__ = ()

    def parse(self, io: IO, context: Context) -> None:
        return None

    def emit(self, value: None, io: IO, context: Context) -> None:
        pass

    def sizeof(self, value: O[None], context: Context) -> O[int]:
        return 0

    def default(self, context: Context) -> None:
        return None
    
    def __repr__(self) -> str:
        return class_name(self)

class Implied(Type):
    __slots__ = ('value',)

    def __init__(self, value: Any = None) -> None:
        self.value = value

    def parse(self, io: IO, context: Context) -> Any:
        return get_value(self.value, context)

    def emit(self, value: Any, io: IO, context: Context) -> None:
        pass

    def sizeof(self, value: O[Any], context: Context) -> int:
        return 0

    def default(self, context: Context) -> Any:
        return peek_value(self.value, context)

    def __repr__(self) -> str:
        return '+{!r}'.format(self.value)

class Ignored(Type, G[T]):
    __slots__ = ('type', 'value')

    def __init__(self, type: T, value: O[Any] = None) -> None:
        self.type = type
        self.value = value

    def parse(self, io: IO, context: Context) -> None:
        parse(self.type, io, context)
        return None

    def emit(self, value: None, io: IO, context: Context) -> None:
        value = get_value(self.value, context)
        if value is None:
            value = default(self.type, context)
        return emit(self.type, value, io, context)

    def sizeof(self, value: None, context: Context) -> int:
        value = peek_value(self.value, context)
        if value is None:
            value = default(self.type, context)
        return _sizeof(self.type, value, context)

    def default(self, context: Context) -> Any:
        return None

    def __repr__(self) -> str:
        return '-{!r}{}'.format(
            class_name(self).strip('<>'),
            '(' + repr(self.value) + ')' if self.value is not None else '',
        )

class Bits(Type):
    __slots__ = ('size', 'byteswap')

    def __init__(self, size: O[int] = None, byteswap: bool = False) -> None:
        self.size = size
        self.byteswap = byteswap

    def parse(self, io: IO, context: Context) -> int:
        size = get_value(self.size, context)
        value = io.read(size, bits=True)
        if get_value(self.byteswap, context):
            value <<= (8 - (size % 8)) % 8
            v = value.to_bytes(math.ceil(size / 8), byteorder='big')
            value = int.from_bytes(v, byteorder='little')
        return value

    def emit(self, value: int, io: IO, context: Context) -> None:
        size = get_value(self.size, context)
        if get_value(self.byteswap, context):
            value = int.from_bytes(value.to_bytes(math.ceil(size / 8), byteorder='big'), byteorder='little')
            value >>= (8 - (size % 8)) % 8
        io.write(value, bits=size)

    def sizeof(self, value: O[int], context: Context) -> O[int]:
        return peek_value(self.size, context) / 8

    def default(self, context: Context) -> int:
        return 0

    def __repr__(self) -> str:
        return '<{}{}>'.format(
            class_name(self),
            ('[' + str(self.size) + ']') if self.size is not None else '',
        )

class Data(Type):
    __slots__ = ('size',)

    def __init__(self, size: O[int] = None) -> None:
        self.size = size

    def parse(self, io: IO, context: Context) -> bytes:
        size = get_value(self.size, context)
        if size is None:
            size = -1
        data = io.read(size)
        if size >= 0 and len(data) != size:
            raise Error(context, 'Size mismatch!\n  wanted: {} bytes\n  found:  {} bytes'.format(
                size, len(data)
            ))
        return data

    def emit(self, value: bytes, io: IO, context: Context) -> None:
        set_value(self.size, len(value), io, context)
        io.write(value)

    def sizeof(self, value: O[bytes], context: Context) -> O[int]:
        if value is not None:
            return len(value)
        return peek_value(self.size, context)

    def default(self, context: Context) -> bytes:
        return b'\x00' * (peek_value(self.size, context) or 0)

    def __repr__(self) -> str:
        return '<{}{}>'.format(
            class_name(self),
            ('[' + str(self.size) + ']') if self.size is not None else '',
        )

class Pad(Type):
    def __new__(self, size: int, value: bytes) -> Ignored[Data]:
        return Ignored(Data(size), value)

class Fixed(Type, G[T]):
    __slots__ = ('pattern', 'type')

    def __init__(self, pattern: Any, type: O[T] = None) -> None:
        self.pattern = pattern
        self.type = type

    def parse(self, io: IO, context: Context) -> bytes:
        pattern = get_value(self.pattern, context)
        type = to_type(self.type or Data(len(pattern)))
        data = parse(type, io, context)
        if data != pattern:
            raise Error(context, 'Value mismatch!\n  wanted: {}\n  found:  {}'.format(
                format_value(pattern, repr), format_value(data, repr),
            ))
        return data

    def emit(self, value: Any, io: IO, context: Context) -> None:
        set_value(self.pattern, value, io, context)
        type = to_type(self.type or Data(len(value)))
        return emit(type, value, io, context)

    def sizeof(self, value: O[Any], context: Context) -> O[int]:
        if value is None:
            value = peek_value(self.pattern, context)
        if self.type is None:
            if value is None:
                return None
            type = Data(len(value))
        else:
            type = to_type(self.type)
        return _sizeof(type, value, context)

    def default(self, context: Context) -> Any:
        return peek_value(self.pattern, context)

    def __repr__(self) -> str:
        if self.type:
            type = to_type(self.type)
            return repr(type) + '!' + repr(self.pattern)
        else:
            return '!' + repr(self.pattern)[1:]


E_co = TypeVar('E_co', bound=enum.Enum)

class Enum(Type, G[E_co, T]):
    __slots__ = ('type', 'cls', 'exhaustive')

    def __init__(self, cls: E_co, type: T, exhaustive: bool = True) -> None:
        self.type = type
        self.cls = cls
        self.exhaustive = exhaustive

    def parse(self, io: IO, context: Context) -> U[E_co, T]:
        value = parse(self.type, io, context)
        try:
            return self.cls(value)
        except ValueError:
            if self.exhaustive:
                raise
            return value

    def emit(self, value: U[E_co, T], io: IO, context: Context) -> None:
        if isinstance(value, self.cls):
            value = value.value
        return emit(self.type, value, io, context)

    def sizeof(self, value: O[U[E_co, T]], context: Context) -> O[int]:
        if value is not None and isinstance(value, self.cls):
            value = value.value
        return _sizeof(self.type, value, context)

    def default(self, context: Context) -> U[E_co, T]:
        return next(iter(self.cls.__members__.values()))

    def __repr__(self) -> str:
        return '<{}({}): {}>'.format(class_name(self), class_name(self.cls), repr(to_type(self.type)).strip('<>'))



## Modifier types

class PartialAttr(Type, G[T]):

    def __init__(self, parent: 'Partial', name: str) -> None:
        self._parent = parent
        self._name = name
        self._type = None
        self._values = []
        self._pvalues = []

    def parse(self, io: IO, context: Context) -> T:
        offset = io.tell()
        value = parse(self._type, io, context)
        for _ in self._parent.types:
            self._values.append((offset, value))
        return value

    def emit(self, value: T, io: IO, context: Context) -> None:
        offset = io.tell()
        for _ in self._parent.types:
            self._values.append((offset, value))
        emit(self._type, value, io, context)

    def sizeof(self, value: O[T], context: Context) -> O[int]:
        for _ in self._parent.types:
            self._pvalues.append(value)
        return _sizeof(self._type, value, context)

    def default(self, context: Context) -> T:
        value = default(self._type, context)
        for _ in self._parent.types:
            self._pvalues.append(value)
        return value

    def get_value(self, context: Context, peek: bool = False) -> T:
        if peek:
            _, value = self._values[-1]
        else:
            _, value = self._values.pop()
        return value

    def peek_value(self, context: Context, default=None) -> T:
        if self._pvalues:
            return self._pvalues.pop()
        if self._values:
            _, value = self._values[-1]
            return value
        return default

    def set_value(self, value: T, io: IO, context: Context) -> None:
        offset, _ = self._values.pop()
        with seeking(io, offset, os.SEEK_SET) as f:
            emit(self._type, value, f, context)

    def __matmul__(self, type: Any) -> Type:
        if isinstance(type, self.__class__):
            return type.__rmatmul__(self)
        return NotImplemented

    def __rmatmul__(self, type: Any) -> Type:
        if isinstance(type, Type):
            return self(type)
        return NotImplemented

    def __setattr__(self, n, v):
        if not n.startswith('_'):
            return setattr(self._type, n, v)
        return super().__setattr__(n, v)

    def __call__(self, type: Type) -> Type:
        self._type = type
        return self

    def __repr__(self) -> str:
        return '</.{}: {}>'.format(self._name, repr(self._type).strip('<>'))

class Partial:
    __slots__ = ('types', 'attrs')

    def __init__(self):
        self.types = []
        self.attrs = {}

    def __getattr__(self, name: str) -> PartialAttr:
        self.attrs[name] = PartialAttr(self, name)
        return self.attrs[name]

    def __matmul__(self, type: Any) -> Type:
        if isinstance(type, Type):
            return self(type)
        return NotImplemented

    def __rmatmul__(self, type: Any) -> Type:
        if isinstance(type, Type):
            return self(type)
        return NotImplemented

    def __call__(self, type: Type) -> Type:
        type = to_type(type)
        for n, v in self.attrs.items():
            setattr(type, n, v)
        self.types.append(type)
        return type

class Ref(Type, G[T]):
    __slots__ = ('type', 'point', 'reference', 'adjustment', 'stream')

    def __init__(self, type: Type, point: O[int] = None, reference: int = os.SEEK_SET, adjustment: U[int, Stream] = 0, stream: O[Stream] = None) -> None:
        self.type = type
        self.point = point
        self.reference = reference
        self.adjustment = adjustment
        self.stream = stream.name if stream else 'refs'

        if self.reference not in (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END):
            raise ValueError('current reference must be any of [os.SEEK_SET, os.SEEK_CUR, os.SEEK_END]')

    def parse(self, io: IO, context: Context) -> T:
        reference = get_value(self.reference, context)
        adjustment = get_value(self.adjustment, context)
        if isinstance(adjustment, Stream):
            start = context.stream_offset(adjustment)
            if reference == os.SEEK_SET:
                adjustment = start
            elif reference == os.SEEK_CUR:
                adjustment = adjustment.pos
            elif reference == os.SEEK_END:
                adjustment = start + context.stream_size(adjustment)
            reference = os.SEEK_SET
        point = get_value(self.point, context) + adjustment
        with context.enter_stream(self.stream, io, point, reference) as f:
            return parse(self.type, f, context)

    def emit(self, value: T, io: IO, context: Context) -> None:
        reference = get_value(self.reference, context)
        adjustment = get_value(self.adjustment, context)
        if isinstance(adjustment, Stream):
            start = context.stream_offset(adjustment)
            if reference == os.SEEK_SET:
                adjustment = start
            elif reference == os.SEEK_CUR:
                adjustment = adjustment.pos
            elif reference == os.SEEK_END:
                adjustment = start + context.stream_size(adjustment)
            reference = os.SEEK_SET

        with context.enter_stream(self.stream, io) as f:
            point = f.tell()
            emit(self.type, value, f, context)

        if reference == os.SEEK_CUR:
            point -= io.tell()
        elif reference == os.SEEK_END:
            with seeking(io, 0, SEEK_END) as f:
                point -= f.tell()

        set_value(self.point, point - adjustment, io, context)

    def sizeof(self, value: O[T], context: Context) -> O[int]:
        with context.enter_stream(self.stream):
            return _sizeof(self.type, value, context)

    def default(self, context: Context) -> T:
        return default(self.type, context)

    def __repr__(self) -> str:
        return '<&{} @ {}{!r}{})>'.format(
            repr(to_type(self.type)).strip('<>'),
            {os.SEEK_SET: '', os.SEEK_CUR: '+', os.SEEK_END: '-'}[self.reference],
            self.point, ('[' + repr(self.adjustment) + ']' if self.adjustment else ''),
        )

class RebasedFile:
    def __init__(self, file: IO, start: O[int] = None) -> None:
        self._file = file
        self._start = start or file.tell()

    def tell(self) -> int:
        return self._file.tell() - self._start

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> None:
        if whence == os.SEEK_SET:
            offset += self._start
        return self._file.seek(offset, whence)

    def __getattr__(self, n: str) -> Any:
        return getattr(self._file, n)

class Rebased(Type, G[T]):
    __slots__ = ('type', 'base')

    def __init__(self, type: Type, base: O[int] = None):
        self.type = type
        self.base = base

    def parse(self, io: IO, context: Context) -> T:
        base = get_value(self.base, context)
        with io.wrapped(RebasedFile(io.handle, base)) as f:
            return parse(self.type, f, context)

    def emit(self, value: T, io: IO, context: Context) -> None:
        base = get_value(self.base, context)
        with io.wrapped(RebasedFile(io.handle, base)) as f:
            return emit(self.type, value, f, context)

    def sizeof(self, value: O[T], context: Context) -> O[int]:
        return _sizeof(self.type, value, context)

    def default(self, context: Context) -> T:
        return default(self.type, context)

    def __repr__(self) -> str:
        return repr(self.type) + ' @ ' + self.base

class SizedFile:
    def __init__(self, file: IO, limit: int, exact: bool = False) -> None:
        self.root = self._file = file
        self._pos = 0
        self._limit = limit
        self._start = file.tell()

    def read(self, n: int = -1) -> bytes:
        remaining = max(0, self._limit - self._pos)
        if n < 0:
            n = remaining
        n = min(n, remaining)
        self._pos += n
        return self._file.read(n)

    def write(self, data: bytes) -> None:
        remaining = self._limit - self._pos
        if len(data) > remaining:
            raise ValueError('trying to write past limit by {} bytes'.format(len(data) - remaining))
        self._pos += len(data)
        return self._file.write(data)

    def seek(self, offset: int, whence: int) -> None:
        if whence == os.SEEK_SET:
            pos = offset
        elif whence == os.SEEK_CUR:
            pos = self._start + self._pos + offset
        elif whence == os.SEEK_END:
            pos = self._start + self._limit - offset
        if pos < self._start:
            raise OSError(errno.EINVAL, os.strerror(errno.EINVAL), offset)
        self._pos = pos - self._start
        return self._file.seek(pos, os.SEEK_SET)

    def tell(self) -> int:
        return self._start + self._pos

    def __getattr__(self, n: str) -> Any:
        return getattr(self._file, n)

class Sized(Type, G[T]):
    __slots__ = ('type', 'limit', 'exact', 'exact_write')

    def __init__(self, type: Type, limit: O[int] = None, exact: bool = False, exact_write: bool = False) -> None:
        self.type = type
        self.limit = limit
        self.exact = exact
        self.exact_write = exact_write

    def parse(self, io: IO, context: Context) -> T:
        exact = get_value(self.exact, context)
        limit = max(0, get_value(self.limit, context))

        start = io.tell()
        with io.wrapped(SizedFile(io.handle, limit, exact)) as f:
            value = parse(self.type, f, context)

        if exact:
            io.seek(start + limit, os.SEEK_SET)
        return value

    def emit(self, value: T, io: IO, context: Context) -> None:
        exact_write = get_value(self.exact_write, context)
        limit = max(0, get_value(self.limit, context, peek=True))

        start = io.tell()
        handle = SizedFile(io.handle, limit, True) if exact_write else io.handle
        with io.wrapped(handle) as f:
            ret = emit(self.type, value, f, context)

        if exact_write:
            io.seek(start + limit, os.SEEK_SET)
        else:
            limit = io.tell() - start
        set_value(self.limit, limit, io, context)
        return ret

    def sizeof(self, value: O[T], context: Context) -> O[int]:
        limit = peek_value(self.limit, context)
        if self.exact:
            return limit
        size = _sizeof(self.type, value, context)
        if size is None:
            return limit
        if limit is None:
            return size
        return min_sizes(*[size, to_size(limit, context)])

    def default(self, context: Context) -> T:
        return default(self.type, context)

    def __repr__(self) -> str:
        return '<{}: {!r} (limit={})>'.format(class_name(self), to_type(self.type), self.limit)

class AlignTo(Type, G[T]):
    __slots__ = ('type', 'alignment', 'value')

    def __init__(self, type: T, alignment: int, value: bytes = b'\x00') -> None:
        self.type = type
        self.alignment = alignment
        self.value = value

    def parse(self, io: IO, context: Context) -> T:
        value = parse(self.type, io, context)
        adjustment = io.tell() % self.alignment
        if adjustment:
            io.seek(self.alignment - adjustment, os.SEEK_CUR)
        return value

    def emit(self, value: T, io: IO, context: Context) -> None:
        emit(self.type, value, io, context)
        adjustment = io.tell() % self.alignment
        if adjustment:
            io.write(self.value * (self.alignment - adjustment))

    def sizeof(self, value: O[T], context: Context) -> O[int]:
        return None # TODO

    def default(self, context: Context) -> T:
        return default(self.type, context)

    def __repr__(self) -> str:
        return '<{}: {!r} (n={})>'.format(class_name(self), to_type(self.type), self.alignment)

class AlignedTo(Type, G[T]):
    __slots__ = ('child', 'alignment', 'value')

    def __init__(self, child: T, alignment: int, value: bytes = b'\x00') -> None:
        self.child = child
        self.alignment = alignment
        self.value = value

    def parse(self, io: IO, context: Context) -> T:
        adjustment = io.tell() % self.alignment
        if adjustment:
            io.seek(self.alignment - adjustment, os.SEEK_CUR)
        value = parse(self.child, io, context)
        return value

    def emit(self, value: T, io: IO, context: Context) -> None:
        adjustment = io.tell() % self.alignment
        if adjustment:
            io.write(self.value * (self.alignment - adjustment))
        emit(self.child, value, io, context)

    def sizeof(self, value: O[T], context: Context) -> O[int]:
        return None # TODO

    def default(self, context: Context) -> T:
        return default(self.type, context)

    def __repr__(self) -> str:
        return '<{}: {!r} (n={})>'.format(class_name(self), self.child, self.alignment)

class LazyEntry(G[T]):
    __slots__ = ('type', 'io', 'pos', 'context', 'parsed')

    def __init__(self, type: T, io: IO, context: Context) -> None:
        self.type = type
        self.io = io
        self.pos = self.io.tell()
        self.context = context
        self.parsed: O[T] = None

    def __call__(self) -> Any:
        if self.parsed is None:
            with seeking(self.io, self.pos) as f:
                self.parsed = parse(self.type, f, self.context)
        return self.parsed

    def __str__(self) -> str:
        return '~~{}'.format(to_type(self.type))

    def __repr__(self) -> str:
        return '<{}: {!r}>'.format(class_name(self), to_type(self.type))

class Lazy(Type, G[T]):
    __slots__ = ('type', 'size')

    def __init__(self, type: T, size: O[int] = None) -> None:
        self.type = type
        self.size = size

    def parse(self, io: IO, context: Context) -> T:
        size = self.sizeof(None, context)[context.stream_path[-1].name]
        if size is None:
            raise ValueError('lazy type size must be known at parse-time')
        entry = LazyEntry(to_type(self.type), io, context)
        io.seek(size, os.SEEK_CUR)
        return entry
    
    def emit(self, value: O[T], io: IO, context: Context) -> None:
        emit(self.type, value(), io, context)

    def sizeof(self, value: O[T], context: Context) -> O[int]:
        length = peek_value(self.size, context)
        if length is not None:
            return length
        if value is not None:
            value = value()
        return _sizeof(self.type, value, context)

    def default(self, context: Context) -> T:
        return default(self.type, context)

    def __str__(self) -> str:
        return '~{}'.format(to_type(self.type))

    def __repr__(self) -> str:
        return '<{}: {!r}>'.format(class_name(self), to_type(self.type))

class Processed(Type, G[T, T2]):
    __slots__ = ('type', 'do_parse', 'do_emit', 'with_context')

    def __init__(self, type: T, parse: Callable[[T], T2], emit: Callable[[T2], T], with_context: bool = False) -> None:
        self.type = type
        self.do_parse = parse
        self.do_emit = emit
        self.with_context = with_context

    def parse(self, io: IO, context: Context) -> T2:
        value = parse(self.type, io, context)
        if get_value(self.with_context, context):
            return self.do_parse(value, context)
        else:
            return self.do_parse(value)

    def emit(self, value: T2, io: IO, context: Context) -> None:
        if get_value(self.with_context, context):
            raw = self.do_emit(value, context)
        else:
            raw = self.do_emit(value)
        emit(self.type, raw, io, context)

    def sizeof(self, value: O[T2], context: Context) -> O[int]:
        if value is not None:
            if peek_value(self.with_context, context):
                raw = self.do_emit(value, context)
            else:
                raw = self.do_emit(value)
        return _sizeof(self.type, raw, context)

    def default(self, context: Context) -> T:
        value = default(self.type, context)
        if peek_value(self.with_context, context):
            return self.do_parse(value, context)
        else:
            return self.do_parse(value)

    def __repr__(self) -> str:
        return '<λ{}{!r} ->{} <-{}>'.format(
            '+' if self.with_context else '',
            to_type(self.type), self.do_parse.__name__, self.do_emit.__name__
        )

class Checked(Type, G[T]):
    __slots__ = ('type', 'check', 'message')

    def __init__(self, type: T, check: Callable[[T], bool], message: O[str] = None) -> None:
        self.type = type
        self.check = check
        self.message = message or 'check failed'

    def parse(self, io: IO, context: Context) -> T:
        check = get_value(self.check, context)
        value = parse(self.type, io, context)
        if not check(value):
            raise Error(context, self.message.format(value))
        return value

    def emit(self, value: T, io: IO, context: Context) -> None:
        check = get_value(self.check, context)
        if not check(value):
            raise Error(context, self.message.format(value))
        return emit(self.type, value, io, context)

    def sizeof(self, value: O[T], context: Context) -> O[int]:
        check = peek_value(self.check, context)
        if value is not None and not check(value):
            raise Error(context, self.message.format(value))
        return _sizeof(self.type, value, context)

    def default(self, context: Context) -> T:
        return default(self.type, context)

    def __repr__(self) -> str:
        return '<?{!r}: {}>'.format(self.check, repr(self.type).strip('<>'))

class Mapped(Type, G[T]):
    def __new__(self, type: T, mapping: Mapping[T, T2], default: O[Any] = None) -> Processed:
        reverse = {v: k for k, v in mapping.items()}
        if default is not None:
            mapping = collections.defaultdict(lambda: default, mapping)
            reverse = collections.defaultdict(lambda: default, reverse)
        return Processed(type, mapping.__getitem__, reverse.__getitem__)


## Compound types

class Generic(Type):
    __slots__ = ('stack',)

    def __init__(self) -> None:
        self.stack = []

    def resolve(self, value) -> None:
        if isinstance(value, Generic):
            self.stack.append(value.stack[-1])
        else:
            self.stack.append(value)

    def pop(self) -> None:
        self.stack.pop()

    def __get_restruct_type__(self, ident: Any) -> Type:
        return to_type(self.stack[-1])

    def parse(self, io: IO, context: Context) -> Any:
        if not self.stack:
            raise Error(context, 'unresolved generic')
        return parse(self.stack[-1], io, context)

    def emit(self, value: O[Any], io: IO, context: Context) -> None:
        if not self.stack:
            raise Error(context, 'unresolved generic')
        return emit(self.stack[-1], value, io, context)

    def sizeof(self, value: O[Any], context: Context) -> O[int]:
        if not self.stack:
            return None
        return _sizeof(self.stack[-1], value, context)

    def default(self, context: Context) -> Any:
        if not self.stack:
            raise Error(context, 'unresolved generic')
        return default(self.stack[-1], context)

    def get_value(self, context: Context, peek: bool = False) -> Any:
        return self.stack[-1]

    def peek_value(self, context: Context) -> Any:
        return self.stack[-1]

    def __repr__(self) -> str:
        if self.stack:
            return '<${}>'.format(repr(to_type(self.stack[-1])).strip('<>'))
        return '<$unresolved>'

    def __deepcopy__(self, memo: Any) -> Any:
        return self

class GenericSelf(Generic):
    def __repr__(self) -> str:
        return '<recursion: {}>'.format(friendly_name(self.stack[-1]))

class MetaSpec(collections.OrderedDict):
    def __getattr__(self, item: Any) -> Any:
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, item: Any, value: Any) -> Any:
        if '__' in item:
            super().__setattr__(item, value)
        else:
            self[item] = value

class StructType(Type):
    __slots__ = ('fields', 'cls', 'generics', 'union', 'partial', 'bound')

    def __init__(self, fields: Mapping[str, Type], cls: type, generics: Sequence[Generic] = [], union: bool = False, partial: bool = False, bound: Sequence[Any] = []) -> None:
        self.fields = MetaSpec(fields)
        self.cls = cls
        self.generics = generics
        self.union = union
        self.partial = partial
        self.bound = bound or []

    def __getitem__(self, ty) -> Any:
        if not isinstance(ty, tuple):
            ty = (ty,)

        bound = self.bound[:]
        bound.extend(ty)
        if len(bound) > len(self.generics):
            raise TypeError('too many generics arguments for {}: {}'.format(
                self.__class__.__name__, len(bound)
            ))

        subtype = self.__class__(self.fields, self.cls, self.generics, self.union, self.partial, bound=bound)
        for i, b in enumerate(subtype.bound):
            # re-bind Self bound
            if b is self:
                subtype.bound[i] = subtype
        return subtype

    def parse(self, io: IO, context: Context) -> Any:
        n = 0
        pos = io.tell()

        c = self.cls.__new__(self.cls)
        try:
            with self.enter():
                for name, type in self.fields.items():
                    with context.enter(name, type):
                        if type is None:
                            continue
                        if self.union:
                            io.seek(pos, os.SEEK_SET)

                        val = parse(type, io, context)

                        nbytes = io.tell() - pos
                        if self.union:
                            n = max(n, nbytes)
                        else:
                            n = nbytes

                        setattr(c, name, val)
                        hook = 'on_parse_' + name
                        if hasattr(c, hook):
                            getattr(c, hook)(self.fields, context)
        except Exception:
            # Check EOF and allow if partial.
            b = io.read(1)
            if not self.partial or b:
                if b:
                    io.seek(-1, os.SEEK_CUR)
                raise
            # allow EOF if partial

        io.seek(pos + n, os.SEEK_SET)
        return c

    def emit(self, value: Any, io: IO, context: Context) -> None:
        n = 0
        pos = io.tell()

        with self.enter():
            for name, type in self.fields.items():
                with context.enter(name, type):
                    if self.union:
                        io.seek(pos, os.SEEK_SET)

                    hook = 'on_emit_' + name
                    if hasattr(value, hook):
                        getattr(value, hook)(self.fields, context)

                    field = getattr(value, name)
                    emit(type, field, io, context)

                    nbytes = io.tell() - pos
                    if self.union:
                        n = max(n, nbytes)
                    else:
                        n = nbytes

        io.seek(pos + n, os.SEEK_SET)

    def sizeof(self, value: O[Any], context: Context) -> O[int]:
        n = {}

        with self.enter():
            for name, type in self.fields.items():
                with context.enter(name, type):
                    if value:
                        field = getattr(value, name)
                    else:
                        field = None

                    nbytes = _sizeof(type, field, context)
                    if nbytes is None:
                        n = None
                        break

                    if self.union:
                        n = max_sizes(n, nbytes)
                    else:
                        n = add_sizes(n, nbytes)

        n = ceil_sizes(n)
        return n

    @contextmanager
    def enter(self):
        for g, child in zip(self.generics, self.bound):
            g.resolve(child)
        yield
        for g in self.generics:
            g.pop()

    def default(self, context: Context) -> Any:
        with self.enter():
            c = self.cls.__new__(self.cls)
            for name, type in self.fields.items():
                with context.enter(name, type):
                    setattr(c, name, default(type, context))

        return c

    def __str__(self) -> str:
        return class_name(self.cls)

    def __repr__(self) -> str:
        type = 'Union' if self.union else 'Struct'
        if self.fields:
            with self.enter():
                fields = '{\n'
                for f, v in self.fields.items():
                    fields += '  ' + f + ': ' + indent(format_value(to_type(v), repr), 2) + ',\n'
                fields += '}'
        else:
            fields = '{}'
        return '<{}({}) {}>'.format(type, class_name(self.cls), fields)

class MetaStruct(type):
    @classmethod
    def __prepare__(mcls, name: str, bases: Sequence[Any], generics: Sequence[str] = [], partials: Sequence[str] = [], inject: bool = True, recursive: bool = False, **kwargs) -> dict:
        attrs = collections.OrderedDict()
        attrs.update({g: Generic() for g in generics})
        attrs.update({p: Partial() for p in partials})
        if inject:
            attrs.update({c.__name__: c for c in __all_types__})
        if recursive:
            attrs['Self'] = GenericSelf()
        return attrs

    def __new__(mcls, name: str, bases: Sequence[Any], attrs: Mapping[str, Any], generics: Sequence[str] = [], partials: Sequence[str] = [], inject: bool = True, recursive: bool = False, **kwargs) -> Any:
        # Inherit some properties from base types
        gs = []
        bound = []
        if recursive:
            gs.append(attrs.pop('Self'))

        fields = {}
        for b in bases:
            fields.update(getattr(b, '__annotations__', {}))
            type = b.__restruct_type__
            gs.extend(type.generics)
            bound.extend(type.bound)
            if type.union:
                kwargs['union'] = True

        for p in partials:
            del attrs[p]
        for g in generics:
            gs.append(attrs.pop(g))
        if inject:
            for c in __all_types__:
                del attrs[c.__name__]
        fields.update(attrs.get('__annotations__', {}))

        attrs['__slots__'] = attrs.get('__slots__', ()) + tuple(fields)

        c = super().__new__(mcls, name, bases, attrs)
        type = StructType(fields, c, gs, bound=bound, **kwargs)
        if recursive:
            type.bound.insert(0, type)
        c.__restruct_type__ = type
        return c

    def __init__(cls, *args, **kwargs) -> Any:
        return super().__init__(*args)

    def __getitem__(cls, ty) -> Any:
        if not isinstance(ty, tuple):
            ty = (ty,)
        subtype = cls.__restruct_type__[ty]
        new_name = '{}[{}]'.format(cls.__name__, ', '.join(friendly_name(r).strip('<>') for r in subtype.bound if r is not subtype))
        new = type(new_name, (cls,), cls.__class__.__prepare__(new_name, (cls,)))
        new.__restruct_type__ = subtype
        new.__slots__ = cls.__slots__
        new.__module__ = cls.__module__
        subtype.cls = new
        return new

    def __repr__(cls) -> str:
        return '<{}: {}>'.format(
            'union' if cls.__restruct_type__.union else 'struct',
            class_name(cls),
        )

class Struct(metaclass=MetaStruct, inject=False):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        st = to_type(self)
        with st.enter():
            for k, t in st.fields.items():
                if k not in kwargs:
                    v = default(t)
                else:
                    v = kwargs.pop(k)
                setattr(self, k, v)

        if kwargs:
            raise TypeError('unrecognized fields: {}'.format(', '.join(kwargs)))

    def __iter__(self) -> Sequence[Any]:
        return iter(self.__slots__)

    def __hash__(self) -> int:
        return hash(tuple((k, getattr(self, k)) for k in self))

    def __eq__(self, other) -> bool:
        if type(self) != type(other):
            return False
        if self.__slots__ != other.__slots__:
            return False
        for k in self:
            ov = getattr(self, k)
            tv = getattr(other, k)
            if ov != tv:
                return False
        return True

    def __fmt__(self, fieldfunc: Callable[[Any], str]) -> str:
        args = []
        for k in self:
            if k.startswith('_'):
                continue
            val = getattr(self, k)
            val = format_value(val, fieldfunc, 2)
            args.append('  {}: {}'.format(k, val))
        args = ',\n'.join(args)
        # Format final value.
        if args:
            return '{} {{\n{}\n}}'.format(class_name(self), args)
        else:
            return '{} {{}}'.format(class_name(self))

    def __str__(self) -> str:
        return self.__fmt__(str)

    def __repr__(self) -> str:
        return self.__fmt__(repr)

class Union(Struct, metaclass=MetaStruct, union=True, inject=False):
    def __setattr__(self, name, value) -> None:
        super().__setattr__(name, value)

        io = BytesIO()
        t = to_type(self)
        try:
            emit(t.fields[name], value, io=io)
        except:
            return

        for fname, ftype in t.fields.items():
            if fname == name:
                continue
            io.seek(0)
            try:
                fvalue = parse(ftype, io)
                super().__setattr__(fname, fvalue)
            except:
                pass

class Tuple(Type):
    __slots__ = ('types',)

    def __init__(self, types: Sequence[Type]) -> None:
        self.types = types

    def parse(self, io: IO, context: Context) -> Sequence[Any]:
        value = []
        for i, type in enumerate(self.types):
            type = to_type(type, i)
            with context.enter(i, type):
                value.append(parse(type, io, context))
        return tuple(value)

    def emit(self, value: Sequence[Any], io: IO, context: Context) -> None:
        for i, (type, val) in enumerate(zip(self.types, value)):
            type = to_type(type, i)
            with context.enter(i, type):
                emit(type, val, io, context)

    def sizeof(self, value: O[Sequence[Any]], context: Context) -> O[int]:
        l = []
        if value is None:
            value = [None] * len(self.types)

        for i, (type, val) in enumerate(zip(self.types, value)):
            type = to_type(type, i)
            with context.enter(i, type):
                l.append(_sizeof(type, val, context))

        return ceil_sizes(add_sizes(*l))

    def default(self, context: Context) -> Sequence[Any]:
        value = []
        for i, type in enumerate(self.types):
            type = to_type(type, i)
            with context.enter(i, type):
                value.append(default(type, context))
        return tuple(value)

    def __repr__(self) -> str:
        return '<(' + ', '.join(repr(to_type(t)) for t in self.types) + ')>'


class Any(Type):
    __slots__ = ('types')

    def __init__(self, types: Sequence[Type]) -> None:
        self.types = types

    def parse(self, io: IO, context: Context) -> Any:
        errors = []
        types = []
        start = io.tell()

        for i, type in enumerate(self.types):
            io.seek(start, os.SEEK_SET)
            type = to_type(type, i)
            with context.enter(i, type):
                try:
                    return parse(type, io, context)
                except Exception as e:
                    if isinstance(e, Error):
                        e = e.exception
                    types.append(type)
                    errors.append(e)

        raise Error(context, 'Failed to parse using any of the following:\n' + '\n'.join(
            ' - {!r} => {}: {}'.format(t, class_name(e), indent(str(e), 2))
            for (t, e) in zip(types, errors)
        ))

    def emit(self, value: Any, io: IO, context: Context) -> None:
        errors = []
        types = []
        start = io.tell()

        for i, type in enumerate(self.types):
            io.seek(start, os.SEEK_SET)
            type = to_type(type, i)
            with context.enter(i, type):
                try:
                    return emit(type, val, io, context)
                except Exception as e:
                    types.append(type)
                    errors.append(e)

        raise Error(context, 'Failed to emit using any of the following:\n' + '\n'.join(
            ' - {!r} => {}: {}'.format(t, class_name(e), indent(str(e), 2))
            for (t, e) in zip(types, errors)
        ))

    def sizeof(self, value: O[Any], context: Context) -> O[int]:
        return None

    def default(self, context: Context) -> O[Any]:
        return None

    def __str__(self) -> str:
        return 'Any[' + ', '.join(format_value(to_type(t, i), str) for i, t in enumerate(self.types)) + ']'

    def __repr__(self) -> str:
        return '<Any[' + ', '.join(format_value(to_type(t, i), repr) for i, t in enumerate(self.types)) + ']>'


class Arr(Type, G[T]):
    __slots__ = ('type', 'count', 'stop_value')

    def __init__(self, type: T, count: O[int] = None, stop_value: O[Any] = None) -> None:
        self.type = type
        self.count = count
        self.stop_value = stop_value

    def parse(self, io: IO, context: Context) -> Sequence[T]:
        value = []

        count = get_value(self.count, context)
        stop_value = get_value(self.stop_value, context)

        i = 0
        while count is None or i < count:
            if isinstance(self.type, list):
                type = to_type(self.type[i], i)
            else:
                type = to_type(self.type, i)

            with context.enter(i, type):
                pos = io.tell()
                try:
                    elem = parse(type, io, context)
                except Exception:
                    # Check EOF.
                    if io.tell() == pos and not io.read(1):
                        break
                    io.seek(-1, os.SEEK_CUR)
                    raise

                if elem == stop_value:
                    break
            
            value.append(elem)
            i += 1

        return value

    def emit(self, value: Sequence[T], io: IO, context: Context) -> None:
        set_value(self.count, len(value), io, context)
        stop_value = get_value(self.stop_value, context)

        if stop_value is not None:
            value = value + [stop_value]

        start = io.tell()
        for i, elem in enumerate(value):
            if isinstance(self.type, list):
                type = to_type(self.type[i], i)
            else:
                type = to_type(self.type, i)

            with context.enter(i, type):
                emit(type, elem, io, context)

    def sizeof(self, value: O[Sequence[T]], context: Context) -> int:
        if value is None:
            count = peek_value(self.count, context)
        else:
            count = len(value)
        stop_value = peek_value(self.stop_value, context)

        if count is None:
            return None

        l = []
        for i in range(count):
            if isinstance(self.type, list):
                type = to_type(self.type[i], i)
            else:
                type = to_type(self.type, i)
            l.append(_sizeof(type, value[i] if value is not None else None, context))

        if stop_value is not None:
            if isinstance(self.type, list):
                type = to_type(self.type[count], count)
            else:
                type = to_type(self.type, count)
            l.append(_sizeof(type, stop_value, context))

        return ceil_sizes(add_sizes(*l))

    def default(self, context: Context) -> Sequence[T]:
        return []

    def __str__(self) -> str:
        return str(to_type(self.type)) + (('[' + str(self.count) + ']') if self.count is not None else '[]')

    def __repr__(self) -> str:
        return '<{}({!r}{}{})>'.format(
            class_name(self), to_type(self.type),
            ('[' + str(self.count) + ']') if self.count is not None else '',
            (', stop: ' + repr(self.stop_value)) if self.stop_value is not None else '',
        )

class Switch(Type):
    __slots__ = ('options', 'selector', 'default_key', 'fallback')

    def __init__(self, default: O[Any] = None, fallback: O[T] = None, options: Mapping[Any, T] = None) -> None:
        self.options = options or {}
        self.selector = None
        self.default_key = default
        self.fallback = fallback

    def _get(self, sel) -> T:
        if sel not in self.options and not self.fallback:
            raise ValueError('Selector {} is invalid! [options: {}]'.format(
                sel, ', '.join(repr(x) for x in self.options.keys())
            ))
        if sel is not None and sel in self.options:
            return self.options[sel]
        else:
            return self.fallback

    def peek_value(self, context: Context) -> O[T]:
        selector = self.selector
        if selector is not None:
            selector = peek_value(self.selector, context, self.default_key)
        if selector is not None:
            return self._get(selector)
        else:
            return self.fallback

    def get_value(self, context: Context) -> T:
        if self.selector is not None:
            return self._get(get_value(self.selector, context))
        elif self.default_key is not None:
            return self._get(self.default_key)
        elif self.fallback is None:
            raise ValueError('Selector not set!')
        else:
            return self.fallback

    def parse(self, io: IO, context: Context) -> Any:
        return parse(self.get_value(context), io, context)

    def emit(self, value: Any, io: IO, context: Context) -> None:
        return emit(self.get_value(context), value, io, context)

    def sizeof(self, value: O[Any], context: Context) -> O[int]:
        type = self.peek_value(context)
        if type is None:
            return None
        return _sizeof(type, value, context)

    def default(self, context: Context) -> O[Any]:
        type = self.peek_value(context)
        if type is None:
            return None
        return default(type, context)

    def __repr__(self) -> str:
        return '<{}: {}>'.format(class_name(self), ', '.join(repr(k) + ': ' + repr(v) for k, v in self.options.items()))


## Primitive types

class Int(Type):
    __slots__ = ('bits', 'signed', 'order')

    def __init__(self, bits: O[int] = None, order: str = 'le', signed: bool = True) -> None:
        self.bits = bits
        self.signed = signed
        self.order = order
    
    def parse(self, io: IO, context: Context) -> int:
        bits = get_value(self.bits, context)
        order = get_value(self.order, context)
        signed = get_value(self.signed, context)
        if bits is not None:
            bs = io.read(bits // 8)
            if len(bs) != bits // 8:
                raise ValueError('short read')
        else:
            bs = io.read()
        return int.from_bytes(bs, byteorder='little' if order == 'le' else 'big', signed=signed)

    def emit(self, value: int, io: IO, context: Context) -> None:
        bits = get_value(self.bits, context)
        order = get_value(self.order, context)
        signed = get_value(self.signed, context)
        if bits is None:
            bits = 8 * math.ceil(value.bit_length() // 8)
        bs = value.to_bytes(bits // 8, byteorder='little' if order == 'le' else 'big', signed=signed)
        io.write(bs)

    def sizeof(self, value: O[int], context: Context) -> int:
        bits = peek_value(self.bits, context)
        if bits is None:
            return None
        return bits // 8

    def default(self, context: Context) -> int:
        return 0

    def __repr__(self) -> str:
        return '<{}{}({}, {})>'.format(
            'U' if not self.signed else '',
            class_name(self), self.bits, self.order
        )

class UInt(Type):
    def __new__(cls, *args, **kwargs) -> Int:
        return Int(*args, signed=False, **kwargs)

class Bool(Type, G[T]):
    def __new__(self, type: T = UInt(8), true_value: T = 1, false_value: T = 0) -> Mapped:
        return Mapped(type, {true_value: True, false_value: False})

class Float(Type):
    __slots__ = ('bits',)

    FORMATS = {
        32: 'f',
        64: 'd',
    }

    def __init__(self, bits: int = 32) -> None:
        self.bits = bits
        if self.bits not in self.FORMATS:
            raise ValueError('unsupported bit count for float: {}'.format(bits))

    def parse(self, io: IO, context: Context) -> float:
        bits = get_value(self.bits, context)
        bs = io.read(bits // 8)
        return struct.unpack(self.FORMATS[bits], bs)[0]

    def emit(self, value: float, io: IO, context: Context) -> None:
        bits = get_value(self.bits, context)
        bs = struct.pack(self.FORMATS[bits], value)
        io.write(bs)

    def sizeof(self, value: O[int], context: Context) -> int:
        bits = peek_value(self.bits, context)
        if bits is None:
            return None
        return bits // 8

    def default(self, context: Context) -> float:
        return 0.0

    def __repr__(self) -> str:
        return '<{}({})>'.format(class_name(self), self.bits)

class Str(Type):
    __slots__ = ('length', 'type', 'encoding', 'terminator', 'exact', 'length_type', 'length_unit')

    def __init__(self, length: O[int] = None, type: str = 'c', encoding: str = 'utf-8', terminator: O[bytes] = None, exact: bool = False, length_unit: int = 1, length_type: Type = UInt(8)) -> None:
        self.length = length
        self.type = type
        self.encoding = encoding
        self.terminator = terminator or b'\x00' * length_unit
        self.exact = exact
        self.length_unit = length_unit
        self.length_type = length_type

        if self.type not in ('raw', 'c', 'pascal'):
            raise ValueError('string type must be any of [raw, c, pascal]')

    def parse(self, io: IO, context: Context) -> str:
        length = get_value(self.length, context)
        length_unit = get_value(self.length_unit, context)
        type = get_value(self.type, context)
        exact = get_value(self.exact, context)
        encoding = get_value(self.encoding,  context)
        terminator = get_value(self.terminator, context)

        if type == 'pascal':
            read_length = parse(self.length_type, io, context)
            if length is not None:
                read_length = min(read_length, length)
            raw = io.read(read_length * length_unit)
        elif type in ('raw', 'c'):
            read_length = 0
            raw = bytearray()
            for i in itertools.count(start=1):
                if length is not None and i > length:
                    break
                c = io.read(length_unit)
                read_length += 1
                if not c or (type == 'c' and c == terminator):
                    break
                raw.extend(c)

        if exact and length is not None:
            if read_length > length:
                raise ValueError('exact length specified but read length ({}) > given length ({})'.format(read_length, length))
            left = length - read_length
            if exact and left:
                io.read(left * length_unit)

        return raw.decode(encoding)

    def emit(self, value: str, io: IO, context: Context) -> None:
        length = get_value(self.length, context)
        length_unit = get_value(self.length_unit, context)
        type = get_value(self.type, context)
        exact = get_value(self.exact, context)
        encoding = get_value(self.encoding,  context)
        terminator = get_value(self.terminator, context)

        raw = value.encode(encoding)

        write_length = (len(value) + (len(terminator) if type == 'c' else 0)) // length_unit
        if type == 'pascal':
            emit(self.length_type, write_length, io, context)
            io.write(raw)
        elif type in ('c', 'raw'):
            io.write(raw)
            if type == 'c':
                io.write(terminator)
        
        if length is not None:
            if write_length > length:
                raise ValueError('exact length specified but write length ({}) > given length ({})'.format(write_length, length))
            left = length - write_length
            if exact and left:
                io.write(b'\x00' * (left * length_unit))

        if not exact:
            set_value(self.length, write_length, io, context)

    def sizeof(self, value: O[str], context: Context) -> O[int]:
        length = peek_value(self.length, context)
        length_unit = peek_value(self.length_unit, context)
        type = peek_value(self.type, context)
        exact = peek_value(self.exact, context)
        encoding = peek_value(self.encoding,  context)
        terminator = peek_value(self.terminator, context)

        if exact and length is not None:
            l = length * length_unit
        elif value is not None:
            l = len(value.encode(encoding))
            if type == 'c':
                l += len(terminator)
        else:
            return None

        if type == 'pascal':
            size_len = _sizeof(self.length_type, l, context)
            if size_len is None:
                return None
            l = add_sizes(to_size(l, context), size_len)

        return l

    def default(self, context: Context) -> str:
        return ''

    def __repr__(self) -> str:
        return '<{}{}({}{})>'.format(self.type.capitalize(), class_name(self), '=' if self.exact else '', self.length)



## Main functions

def to_io(value: Any) -> IO:
    if isinstance(value, IO):
        return value
    if value is None:
        value = BytesIO()
    if isinstance(value, (bytes, bytearray)):
        value = BytesIO(value)
    return IO(value)

def to_type(spec: Any, ident: O[Any] = None) -> Type:
    if isinstance(spec, Type):
        return spec
    if isinstance(spec, (list, tuple)):
        return Tuple(spec)
    elif hasattr(spec, '__restruct_type__'):
        return spec.__restruct_type__
    elif hasattr(spec, '__get_restruct_type__'):
        return spec.__get_restruct_type__(ident)
    elif callable(spec):
        return spec(ident)

    raise ValueError('Could not figure out specification from argument {}.'.format(spec))

def get_value(t: Type, context: Context, peek: bool = False) -> Any:
    if isinstance(t, (Generic, PartialAttr)):
        return t.get_value(context, peek)
    return t

def peek_value(t: Type, context: Context, default=None) -> Any:
    if isinstance(t, (Generic, PartialAttr)):
        return t.peek_value(context, default)
    return t

def set_value(t: Type, value: Any, io: IO, context: Context) -> None:
    if isinstance(t, PartialAttr):
        t.set_value(value, io, context)

def to_size(v: Any, context: Context) -> Mapping[str, int]:
    if not isinstance(v, dict):
        stream = context.stream_path[-1] if context.stream_path else context.params.default_stream
        v = {stream.name: v}
    return v

def parse(spec: Any, io: IO, context: O[Context] = None, params: O[Params] = None) -> Any:
    type = to_type(spec)
    io = to_io(io)
    context = context or Context(type, params=params)
    at_start = not context.path
    try:
        return type.parse(io, context)
    except Error:
        raise
    except Exception as e:
        if at_start:
            raise Error(context, e)
        else:
            raise

def emit(spec: Any, value: Any, io: O[IO] = None, context: O[Context] = None, params: O[Params] = None) -> None:
    type = to_type(spec)
    io = to_io(io)
    ctx = context or Context(type, value, params=params)
    try:
        type.emit(value, io, ctx)
        return io.handle
    except Error:
        raise
    except Exception as e:
        if not context:
            raise Error(ctx, e)
        else:
            raise

def _sizeof(spec: Any, value: O[Any], context: Context) -> Mapping[str, O[int]]:
    type = to_type(spec)
    return to_size(type.sizeof(value, context), context)

def sizeof(spec: Any, value: O[Any] = None, context: O[Context] = None, params: O[Params] = None, stream: O[Str] = None) -> O[int]:
    type = to_type(spec)
    ctx = context or Context(type, value, params=params)
    try:
        sizes = _sizeof(type, value, ctx)
    except Exception as e:
        raise Error(ctx, e)

    if stream:
        return sizes.get(stream, 0)
    else:
        n = 0
        for v in sizes.values():
            if v is None:
                return None
            n += v
        return n

def default(spec: Any, context: O[Context] = None, params: O[Params] = None) -> O[Any]:
    type = to_type(spec)
    ctx = context or Context(type, params=params)
    try:
        return type.default(ctx)
    except Error:
        raise
    except Exception as e:
        if not context:
            raise Error(ctx, e)
        else:
            raise


__all_types__ = {
    # Base types
    Nothing, Bits, Data, Implied, Ignored, Pad, Fixed, Generic,
    # Modifier types
    Ref, Rebased, Sized, AlignTo, AlignedTo, Lazy, Processed, Checked, Mapped,
    # Compound types
    StructType, MetaStruct, Struct, Union, Tuple, Any, Arr, Switch, Enum,
    # Primitive types
    Bool, Int, UInt, Float, Str,
}
__all__ = [c.__name__ for c in __all_types__ | {
    # Bases
    IO, Context, Params, Error, Type,
    # Functions
    parse, emit, sizeof, default,
}]

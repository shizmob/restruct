restruct
========

Declarative binary file format parser and emitter for Python 3.

Quickstart
----------

```python3
from restruct import parse, emit, UInt, Arr

# Parse a simple little-endian integer
>>> parse(UInt(32), b'\x39\x05\x00\x00')
1337

# Parse an array that continues 'til end-of-stream
>>> parse(Arr(UInt(8)), b'\x45\x45\x45\x45')
[69, 69, 69, 69]

# Parse any bytes-like structures and files!
>>> parse(Arr(UInt(8)), open('foo.bin', 'rb'))
[13, 37]

# Emit data out again!
>>> emit(UInt(32), 420, open('bar.bin', 'wb'))

# Or emit to a BytesIO object if none given
>>> emit(Arr(UInt(8)), [13, 37, 69, 69])
<_io.BytesIO object at 0x106cc0810>
>>> _.getvalue().hex()
'0d254545'
```

Structures
----------

```python3
from restruct import parse, Struct

class Test(Struct):
    # restruct standard types are injected by default, so no need to import them
    foo: UInt(32)
    bar: Str(type='c')

>>> parse(Test, b'\x39\x05\x00\x00Hello world!\x00Garbage')
Test {
  foo: 1337,
  bar: 'Hello world!'
}
>>> _.foo
1337
```

Hooks
-----

```python3
from restruct import parse, Struct

class Stats(Struct):
    engine_level: UInt(32)
    rpm:          UInt(16)

class Message(Struct):
    message:  Str(length=64, exact=True)
    priority: UInt(8)

class Test(Struct):
    type:     UInt(32)
    contents: Switch(options={
        1: Stats,
        2: Message,
    })

    def on_type(self, spec, context):
        # called when `type` field is set, spec contains the field types
        spec.contents.selector = self.type

>>> parse(Test, b"\x02\x00\x00\x00Did you expect a cute foo? Too bad, it's just me, bar!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x63")
Test {
  type: 2,
  contents: Message {
    message: Did you expect a cute foo? Too bad, it's just me, bar!,
    priority: 99
  }
}
```

Inheritance
-----------

```python3
from restruct import parse, Struct

class Base(Struct):
    a: UInt(8)
    b: UInt(8)

class Derived(Base):
    c: UInt(8)
    d: UInt(8)

# just works!
>>> parse(Derived, b'\x01\x02\x03\x04')
Derived {
  a: 1,
  b: 2,
  c: 3,
  d: 4
}
```

Generics
--------

```python3
from restruct import parse, Struct, UInt

class GenericTest(Struct, generics=['T']):
    # now you can use the variable T to stand in for any type and most values!
    foo: UInt(32)
    bar: Arr(T)

# use [] syntax on the type to resolve the generic
>>> parse(GenericTest[UInt(16)], b'\x39\x05\x00\x00\x45\x00\xa4\x01\x11\x22')
GenericTest[UInt(16, le)] {
  foo: 1337,
  bar: [69, 420, 8721]
}

# failing to resolve all generics before parsing predictably fails
>>> parse(GenericTest, b'\x39\x05\x00\x00huh?')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "restruct.py", line 1225, in parse
    return type.parse(io, context)
  File "restruct.py", line 695, in parse
    val = parse(type, io, context)
  File "restruct.py", line 1225, in parse
    return type.parse(io, context)
  File "restruct.py", line 619, in parse
    raise Error(context, 'unresolved generic')
restruct.Error: [bar] ValueError: unresolved generic

# also works with inheritance!
```

Error handling
--------------

```python3
from restruct import parse, Struct

class Inner(Struct):
    foo: Str(length=32, type='c')

class Nested(Struct):
    level: UInt(8)
    inner: Arr(Inner, count=4)

class Base(Struct):
    version: UInt(8)
    nested:  Nested

# errors contain the full path through the structures to the error'd value
>>> parse(Base, b'\x01\x45All\x00Good\x00So\x00\x81hmm\x00\x00')
Traceback (most recent call last):
  File "restruct.py", line 1225, in parse
    return type.parse(io, context)
  File "restruct.py", line 695, in parse
    val = parse(type, io, context)
  File "restruct.py", line 1225, in parse
    return type.parse(io, context)
  File "restruct.py", line 695, in parse
    val = parse(type, io, context)
  File "restruct.py", line 1225, in parse
    return type.parse(io, context)
  File "restruct.py", line 907, in parse
    elem = parse(type, io, context)
  File "restruct.py", line 1225, in parse
    return type.parse(io, context)
  File "restruct.py", line 695, in parse
    val = parse(type, io, context)
  File "restruct.py", line 1225, in parse
    return type.parse(io, context)
  File "restruct.py", line 1133, in parse
    return raw.decode(encoding)
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x81 in position 0: invalid start byte

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "restruct.py", line 1230, in parse
    raise Error(context, e)
restruct.Error: [nested.inner[3].foo] UnicodeDecodeError: 'utf-8' codec can't decode byte 0x81 in position 0: invalid start byte
# access the path programmatically
>>> e.path
[('nested', <struct: Nested>), ('inner', <Arr(<struct: Inner>[4])>), (3, <restruct.StructType object at 0x1079b2810>), ('foo', <CStr(32)>)]
# access the original exception
>>> e.exception
UnicodeDecodeError('utf-8', b'\x81', 0, 1, 'invalid start byte')
```

Standard types
--------------

* `Int(bits, order='le', signed=True):` two's-complement integer
* `UInt(bits, order='le'):` two's-complement unsigned integer
* `Float(bits):` IEEE754 binary float
* `Str(length?=None, type='c', encoding='utf-8', terminator?=None, exact=False, length_unit=1, length_type=UInt(8)):` string, supported types are `raw`, `c` and `pascal`
* `Bool(type=UInt(8), true_value=1, false_value=0):` generic boolean

---

* `Nothing:` parses nothing and emits nothing, returns `None`
* `Implied(value):` parses nothing and emits nothing, returns `value`
* `Fixed(value):` reads bytes and emits bytes, making sure they equate `value`
* `Pad(size, value?=b'\x00'):` parses and emits padding bytes, returns `None`
* `Data(size?=None):` parses and returns raw bytes
* `Enum(enum, type):` parses and emits `type` and constructs `enum.Enum` subclass `enum` with its result

---

* `StructType(fields, cls, generics=[], union=False, partial=False, bound=[]):` type class used by `MetaStruct`
* `Struct:` base class for automatic struct type generation through `MetaStruct` meta-class and field annotations
* `Union:` base class for automatic union type generation through `MetaStruct` meta-class and field annotations
* `Arr(type, count=None, size=None, stop_value=None):` parses and emits array of `types`, of optionally max `count` elements and `size` bytes total size
* `Switch(default=None, fallback=None, options={}):` parses and emits a choice of types chosen through the `selector` field

---

* `AtOffset(type, point=None, reference=os.SEEK_SET):` parses and emits `type` at offset `point` in input stream
* `Ref(value_type, offset_type, reference=os.SEEK_SET):` parses and emits `value_type` at offset `offset_type`, parsed before, in the stream
* `WithSize(type, limit=None, exact=False):` parses and emits `type`, limiting its size in the tream to `limit` bytes
* `AlignTo(type, alignment, value?=b'\x00'):` parses and emits `type`, aligning stream to alignment bytes **after**
* `AlignedTo(type, alignment, value?=b'\x00'):` parse and emits `type`, aligning stream to alignment bytes **before**
* `Lazy(type, size):` parses and emits `type` lazily, returning a callable that will parse and return the type when
* `Processed(type, parse, emit):` parses and emits `type`, processing them through `parse` and `emit` callables, respectively
* `Mapped(type, mapping, default?=None):` parses and emits `type`, looking up the result in `mapping`

License
=======

BSD-2; see `LICENSE` for details.

TODO
====

* Properly emit `AtOffset` and `Ref` types
* Add `Maybe` and `Either` types
* Fix `Arr` EOF-handling masking errors
* Port more features over from `destruct`
* More examples

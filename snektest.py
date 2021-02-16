
from snek import setup
from myia import abstract as A
from myia import lib
from myia import xtype
from myia.pipeline.standard import standard_parse as parse


setup()

print(lib.to_abstract((1, 2, 3)))
print(lib.to_abstract([1, 2, 3]))


@parse
def f(x):
    return x * x

print(f)
print(f.nodes)

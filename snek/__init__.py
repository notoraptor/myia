import snektalk
from hrepr import hrepr
from .repr import MyiaHrepr
from snektalk import SnekTalkDb, interact, debug

def setup():
    hrepr.configure(
        mixins=MyiaHrepr,
    )

setup()

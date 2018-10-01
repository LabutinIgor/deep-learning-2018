from resnext.resnext import *
from resnext.train import *


def test_resnext():
    net = resnext50()
    assert net is not None
    assert False

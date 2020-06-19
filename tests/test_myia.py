import numpy as np
import pytest

from myia.frontends import activate_frontend  # noqa: E402
from .common import af32_of
from .multitest import eqtest, infer, mt, run_relay as run

torch = pytest.importorskip("torch")

activate_frontend("pytorch")


@eqtest.register
def eqtest(t1: torch.Tensor, t2, rtol=1e-5, atol=1e-8, **kwargs):
    """ New version of eqtest using np.testing.assert_allclose.
    If comparison fails, this version will raise an exception
    and display a more informative log if comparison fail,
    especially max absolute and relative difference.
    """
    np.testing.assert_allclose(
        t1.detach().numpy(),
        t2.detach().numpy(),
        rtol=rtol,
        atol=atol,
        verbose=True,
    )
    return True


@mt(
    run(1, 2, result=3),
    run(-1, -3, result=-4),
    infer(af32_of(2, 3, 4), af32_of(2, 3, 4), result=af32_of(2, 3, 4)),
    run(np.random.randn(2, 3, 4), np.random.randn(2, 3, 4)),
)
def test_add(a, b):
    return a + b


@mt(
    run(3, 4, result=12),
    run(-7, 4, result=-28),
    infer(af32_of(2, 3, 4), af32_of(2, 3, 4), result=af32_of(2, 3, 4)),
    run(np.random.randn(2, 3, 4), np.random.randn(2, 3, 4)),
)
def test_mul(a, b):
    return a * b


@mt(
    infer(af32_of(2, 3), af32_of(3, 7), result=af32_of(2, 7)),
    run(np.random.randn(2, 3), np.random.randn(3, 7)),
)
def test_matmul(a, b):
    return a @ b


@mt(run(5, 7, result=5), run(12, -1, result=-1))
def test_if_else(a, b):
    if a < b:
        return a
    else:
        return b


@mt(run(12, 4), run(-5, 9), run(np.random.randn(2, 3), np.random.randn(2, 3)))
def test_sub_function(a, b):
    def fn(a, b):
        return a * b

    return (a + b) * fn(a, b)


@mt(
    run(
        torch.randn(2, 6, 4, 5, dtype=torch.float32),
        torch.randn(3, 2, 3, 3, dtype=torch.float32),
        torch.randn(3, dtype=torch.float32),
    ),
    run(
        torch.randn(2, 3, 4, 5, dtype=torch.float32),
        torch.randn(3, 1, 3, 3, dtype=torch.float32),
        torch.randn(3, dtype=torch.float32),
    ),
    run(
        torch.randn(2, 6, 4, 5, dtype=torch.float32),
        torch.randn(3, 2, 3, 3, dtype=torch.float32),
        None,
    ),
)
def test_torch_conv2d(inp, w, b):
    return torch.nn.functional.conv2d(inp, w, b, (2, 3), (3, 2), (3, 4), 3)

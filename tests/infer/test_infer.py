from myia.testing.multitest import infer, mt

from ..common import A


@mt(
    infer(A(int), A(int), result=A(float)),  # should fail
    infer(A(int), A(int), result=A(int)),
)
def test_sum(a, b):
    return a + b


@infer(A(int), A(int), result=A(bool))  # should fail
def test_sum_2(a, b):
    return a + b

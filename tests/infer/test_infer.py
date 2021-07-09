from myia.testing.multitest import infer, mt

from myia.testing.common import A


# Test `mt`
@mt(
    infer(A(int), A(int), result=A(int)),
    infer(A(float), A(float), result=A(float)),
    infer(int, int, result=int),
    infer(float, float, result=float),
)
def test_sum(a, b):
    return a + b


# Test `infer` alone
@infer(A(int), A(int), result=A(int))
def test_sum_2(a, b):
    return a + b

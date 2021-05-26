import pytest

from myia.ir.node import SEQ, Constant, Graph, Node, Parameter
from myia.utils.info import enable_debug


def test_graph_output():
    g = Graph()

    with pytest.raises(ValueError):
        g.output

    c = g.constant(0)
    g.output = c

    assert g.output.is_constant() and g.output.value == 0

    c = g.constant(2)
    g.output = c

    assert g.output.is_constant() and g.output.value == 2


def test_graph_flags():
    g = Graph()

    assert "core" not in g.flags

    g.set_flags(core=True)

    assert g.flags["core"]

    g.set_flags(core=False)

    assert not g.flags["core"]


def test_parameter():
    g = Graph()

    p = g.add_parameter("a")
    assert p.is_parameter()
    assert p.graph is g
    assert p.name == "a"
    assert g.parameters[0] is p

    p2 = g.add_parameter("b")
    assert g.parameters[1] is p2


def test_apply():
    g = Graph()

    c = g.constant(0)

    a = g.apply("add", c, 1)

    assert a.is_apply()
    assert a.is_apply("add")
    assert a.graph is g
    assert isinstance(a.edges[1].node, Node)

    assert len(a.edges) == 3
    assert a.fn.value == "add"
    assert a.inputs == (c, a.edges[1].node)

    a2 = g.apply("sub", a, c)
    a2.add_edge(SEQ, a)

    assert a2.edges[SEQ].node is a

    with pytest.raises(AssertionError):
        a2.add_edge(SEQ, None)


def test_constant():
    g = Graph()

    c = g.constant(0)

    assert c.value == 0
    assert c.is_constant()
    assert not c.is_constant_graph()

    c2 = g.constant(g)

    assert c2.is_constant_graph()


def test_clone():
    f = Graph()
    a = f.add_parameter("a")
    g = Graph()
    b = g.add_parameter("b")
    f.output = f.apply(g, a)
    g.output = g.apply("fma", b, 0, a)
    g.return_.add_edge(SEQ, g.output)

    g2 = g.clone()

    assert len(g2.parameters) == 1
    assert g2.return_
    assert g2.return_ is not g.return_
    assert g2.output.is_apply("fma")
    assert g2.output is not g.output

    fma2 = g2.output
    fma = g.output
    assert fma2.edges[0].node is not fma.edges[0].node
    assert fma2.edges[1].node is not fma.edges[1].node
    assert fma2.edges[2].node is fma.edges[2].node

    f2 = f.clone()
    f2.output.fn.value is f.output.fn.value

    op = f.apply("op", fma)
    op2 = op.clone(f, {f: f})
    assert op2.edges[0].node is op.edges[0].node

    c = Constant(1)
    op3 = g.apply("op", c, c)
    op4 = op3.clone(g, {g: g})
    assert op3.edges[0].node is not op4.edges[0].node
    assert op4.edges[0].node is op4.edges[1].node


def test_clone_closure():
    g = Graph()
    p = g.add_parameter("p")
    g2 = Graph(parent=g)
    g2.output = g2.apply("add", p, 1)
    g.output = g.apply("mul", g.apply(g2), g.apply(g2))

    h = g.clone()
    m = h.output
    h2 = m.edges[0].node.fn.value
    h2b = m.edges[1].node.fn.value
    assert h2 is not g2
    assert h2 is h2b
    assert h2.output.edges[0].node is h.parameters[0]


def test_graph_replace():
    g = Graph()
    p = g.add_parameter("p")
    a = g.apply("add", p, 0)
    a2 = g.apply("add", p, a)
    a2.add_edge(SEQ, a)
    a3 = g.apply("op", a2, a)
    a3.add_edge(SEQ, a2)
    g.output = a3

    r = {a: p}
    r_s = {a: None}
    g.replace(r, r_s)

    assert SEQ not in g.output.edges[SEQ].node.edges
    assert g.output.edges[1].node is p
    assert g.output.edges[0].node.edges[1].node is p


def test_graph_replace2():
    g = Graph()
    p = g.add_parameter("p")
    a = g.apply("add", p, 0)
    a2 = g.apply("add", p, a)
    a2.add_edge(SEQ, a)
    a3 = g.apply("op", a2, a)
    a3.add_edge(SEQ, a2)
    g.output = a3

    b = g.apply("make_int", p)

    r = {a: b}
    r_s = {a2: b}
    g.replace(r, r_s)

    assert g.output.edges[SEQ].node is b
    assert g.output.edges[1].node is b
    assert g.output.edges[0].node.edges[0].node is p


def test_graph_replace3():
    g = Graph()
    p = g.add_parameter("p")
    a = g.apply("add", p, 0)
    g2 = Graph(parent=g)
    g2.output = g2.apply("add", p, a)
    g.output = g.apply(g2)
    g.output.add_edge(SEQ, a)

    r = {a: p}
    r_s = {a: None}
    g.replace(r, r_s)

    assert g.output.fn.value.output.edges[1].node is p


def test_graph_add_debug():
    g = Graph()

    g.add_debug(name="g", param=1)

    assert g.debug is None

    with enable_debug():
        g2 = Graph()

    g2.add_debug(name="g", param=1)

    assert g2.debug is not None
    assert g2.debug.name == "g"
    assert g2.debug.param == 1


def test_node():
    c = Constant(0)

    assert not c.is_apply()
    assert not c.is_parameter()

    c.add_debug(name="c", value=0)

    assert c.debug is None

    with enable_debug():
        p = Parameter(None, "a")

    assert not p.is_constant()
    assert not p.is_constant_graph()

    p.add_debug(floating=True)

    assert p.debug is not None
    assert p.debug.floating
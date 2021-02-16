import os
from snektalk.utils import Interactor
from hrepr import hrepr

from myia.lib import manage, GraphCloner, Apply, Constant
from myia.opt import LocalPassOptimizer, pattern_replacer
from myia.operations import primitives as primops
from myia.debug.label import (
    CosmeticPrimitive,
    NodeLabeler,
    short_labeler,
    short_relation_symbols,
)
from myia.utils import Registry
from myia.utils.unify import SVar, Var, var
from myia import operations
from myia.info import about


mcss_path = f"{os.path.dirname(__file__)}/assets/myia.css"
mcss = open(mcss_path).read()


gcss_path = f"{os.path.dirname(__file__)}/assets/graph.css"
gcss = open(gcss_path).read()


cosmetics = Registry()


@cosmetics.register(primops.return_)
def _cosmetic_node_return(self, node, g, cl):
    """Create node and edges for `return ...`."""
    self.cynode(id=node, label="", parent=g, classes="const_output")
    ret = node.inputs[1]
    self.process_edges([(node, "", ret)])


class GraphCosmeticPrimitive(CosmeticPrimitive):
    """Cosmetic primitive that prints pretty in graphs.

    Attributes:
        on_edge: Whether to display the label on the edge.

    """

    def __init__(self, label, on_edge=False):
        """Initialize a GraphCosmeticPrimitive."""
        super().__init__(label)
        self.on_edge = on_edge

    def graph_display(self, gprint, node, g, cl):
        """Display a node in cytoscape graph."""
        if gprint.function_in_node and self.on_edge:
            lbl = gprint.label(node, "")
            gprint.cynode(id=node, label=lbl, parent=g, classes=cl)
            gprint.process_edges(
                [(node, (self.label, "fn-edge"), node.inputs[1])]
            )
        else:
            gprint.process_node_generic(node, g, cl)


make_tuple = GraphCosmeticPrimitive("(...)")


X = Var("X")
Y = Var("Y")
Xs = SVar(Var())
V = var(lambda x: x.is_constant())
V1 = var(lambda x: x.is_constant())
V2 = var(lambda x: x.is_constant())
L = var(lambda x: x.is_constant_graph())


@pattern_replacer(primops.make_tuple, Xs)
def _opt_fancy_make_tuple(optimizer, node, equiv):
    xs = equiv[Xs]
    ct = Constant(GraphCosmeticPrimitive("(...)"))
    with about(node.debug, "cosmetic"):
        return Apply([ct, *xs], node.graph)


@pattern_replacer(primops.tuple_getitem, X, V)
def _opt_fancy_getitem(optimizer, node, equiv):
    x = equiv[X]
    v = equiv[V]
    ct = Constant(GraphCosmeticPrimitive(f"[{v.value}]", on_edge=True))
    with about(node.debug, "cosmetic"):
        return Apply([ct, x], node.graph)


@pattern_replacer(operations.resolve, V1, V2)
def _opt_fancy_resolve(optimizer, node, equiv):
    ns = equiv[V1]
    name = equiv[V2]
    with about(node.debug, "cosmetic"):
        lbl = f"{ns.value.label}.{name.value}"
        ct = Constant(GraphCosmeticPrimitive(lbl))
        return ct


@pattern_replacer(primops.record_getitem, X, V)
def _opt_fancy_record_getitem(optimizer, node, equiv):
    x = equiv[X]
    v = equiv[V]
    ct = Constant(GraphCosmeticPrimitive(f"{v.value}", on_edge=True))
    with about(node.debug, "cosmetic"):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.unsafe_static_cast, X, V)
def _opt_fancy_unsafe_static_cast(optimizer, node, equiv):
    x = equiv[X]
    ct = Constant(GraphCosmeticPrimitive(f"cast", on_edge=True))
    with about(node.debug, "cosmetic"):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.hastag, X, V)
def _opt_fancy_hastag(optimizer, node, equiv):
    x = equiv[X]
    v = equiv[V]
    ct = Constant(GraphCosmeticPrimitive(f"?{v.value}", on_edge=True))
    with about(node.debug, "cosmetic"):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.casttag, X, V)
def _opt_fancy_casttag(optimizer, node, equiv):
    x = equiv[X]
    v = equiv[V]
    ct = Constant(GraphCosmeticPrimitive(f"!{v.value}", on_edge=True))
    with about(node.debug, "cosmetic"):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.tagged, X, V)
def _opt_fancy_tagged(optimizer, node, equiv):
    x = equiv[X]
    v = equiv[V]
    ct = Constant(GraphCosmeticPrimitive(f"@{v.value}", on_edge=True))
    with about(node.debug, "cosmetic"):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.array_map, V, Xs)
def _opt_fancy_array_map(optimizer, node, equiv):
    xs = equiv[Xs]
    v = equiv[V]
    if v.is_constant_graph():
        return node
    name = short_labeler.label(v)
    ct = Constant(GraphCosmeticPrimitive(f"[{name}]"))
    with about(node.debug, "cosmetic"):
        return Apply([ct, *xs], node.graph)


@pattern_replacer(primops.distribute, X, V)
def _opt_fancy_distribute(optimizer, node, equiv):
    x = equiv[X]
    v = equiv[V]
    ct = Constant(GraphCosmeticPrimitive(f"shape→{v.value}", on_edge=True))
    with about(node.debug, "cosmetic"):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.scalar_to_array, X, V)
def _opt_fancy_scalar_to_array(optimizer, node, equiv):
    x = equiv[X]
    ct = Constant(GraphCosmeticPrimitive(f"to_array", on_edge=True))
    with about(node.debug, "cosmetic"):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.array_to_scalar, X)
def _opt_fancy_array_to_scalar(optimizer, node, equiv):
    x = equiv[X]
    ct = Constant(GraphCosmeticPrimitive(f"to_scalar", on_edge=True))
    with about(node.debug, "cosmetic"):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.transpose, X, V)
def _opt_fancy_transpose(optimizer, node, equiv):
    if equiv[V].value == (1, 0):
        x = equiv[X]
        ct = Constant(GraphCosmeticPrimitive(f"T", on_edge=True))
        with about(node.debug, "cosmetic"):
            return Apply([ct, x], node.graph)
    else:
        return node


@pattern_replacer(primops.array_reduce, primops.scalar_add, X, V)
def _opt_fancy_sum(optimizer, node, equiv):
    x = equiv[X]
    shp = equiv[V].value
    ct = Constant(GraphCosmeticPrimitive(f'sum {"x".join(map(str, shp))}'))
    with about(node.debug, "cosmetic"):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.distribute, (primops.scalar_to_array, V, V2), X)
def _opt_distributed_constant(optimizer, node, equiv):
    return equiv[V]


def cosmetic_transformer(g):
    """Transform a graph so that it looks nicer.

    The resulting graph is not a valid one to run, because it may contain nodes
    with fake functions that only serve a cosmetic purpose.
    """
    spec = (
        _opt_distributed_constant,
        _opt_fancy_make_tuple,
        _opt_fancy_getitem,
        _opt_fancy_resolve,
        _opt_fancy_record_getitem,
        _opt_fancy_array_map,
        _opt_fancy_distribute,
        _opt_fancy_transpose,
        _opt_fancy_sum,
        _opt_fancy_unsafe_static_cast,
        _opt_fancy_scalar_to_array,
        _opt_fancy_array_to_scalar,
        _opt_fancy_hastag,
        _opt_fancy_casttag,
        _opt_fancy_tagged,
        # careful=True
    )
    optim = LocalPassOptimizer(*spec)
    optim(g)
    return g


class Interactor2(Interactor):

    @classmethod
    def __hrepr_resources__(cls, H):
        reqs = [
            H.javascript(export=name, src=src["url"])
            for name, src in cls.js_requires.items()
        ]
        if cls.js_code:
            main = H.javascript(
                cls.js_code,
                require={k: v["varname"] for k, v in cls.js_requires.items()}
            )
        else:
            main = H.javascript(src=cls.js_source)
        return [*reqs, main(export=cls.js_constructor)]


class CytoscapeGraph(Interactor2):
    js_constructor = "CytoscapeGraph"
    js_requires = {
        "cytoscape": {
            "url": "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.18.0/cytoscape.min.js",
            "varname": "cytoscape"
        },
        "cytoscape-dagre": {
            "url": "https://cdn.rawgit.com/cytoscape/cytoscape.js-dagre/1.5.0/cytoscape-dagre.js",
            "varname": "cydagre"
        },
        "dagre": {
            "url": "https://cdn.rawgit.com/cpettitt/dagre/v0.7.4/dist/dagre.js",
            "varname": "dagre"
        },
        # "cytoscape-popper": {
        #     "url": "https://cdnjs.cloudflare.com/ajax/libs/cytoscape-popper/2.0.0/cytoscape-popper.js",
        #     "varname": "cypopper"
        # },
        # "popper.js": {
        #     "url": "https://cdnjs.cloudflare.com/ajax/libs/popper.js/2.6.0/umd/popper.js",
        #     "varname": "popper"
        # },
        # "@popper/core": {
        #     "url": "https://cdnjs.cloudflare.com/ajax/libs/popper.js/2.6.0/umd/popper.js",
        #     "varname": "popper2"
        # },
        # "tippy": {
        #     "url": "https://cdnjs.cloudflare.com/ajax/libs/tippy.js/2.5.4/tippy.standalone.js",
        #     "varname": "tippy"
        # }
    }
    js_code = """
    cydagre(cytoscape, dagre);

    class CytoscapeGraph {
        constructor(element, options) {
            element.style.height = options.height || "500px";
            element.style.width = options.width || "500px";
            element.onclick = () => {};
            options.container = element;
            this.cy = cytoscape(options);
            if (options.on_node) {
                this.cy.on('click', 'node', function(evt){
                    options.on_node(evt.target.data());
                });
            }
        }
    }
    """

    @classmethod
    def __hrepr_resources__(cls, H):
        res = super(CytoscapeGraph, cls).__hrepr_resources__(H)
        return [H.style(mcss), *res]


class GraphPrinter:
    """Utility to generate a graphical representation for a graph.

    This is intended to be used as a base class for classes that
    specialize over particular graph structures.

    """

    def __init__(self, cyoptions, tooltip_gen=None, extra_style=None, on_node=None):
        """Initialize GraphPrinter."""
        # Nodes and edges are accumulated in these lists
        self.nodes = []
        self.edges = []
        self.cyoptions = cyoptions
        self.tooltip_gen = tooltip_gen
        self.extra_style = extra_style or ""
        self.id_to_obj = {}
        if on_node is not None:
            def _on_node(data):
                return on_node(self.id_to_obj[data["id"]])
            self.on_node = _on_node
        else:
            self.on_node = None

    def id(self, x):
        """Return the id associated to x."""
        rval = f"X{id(x)}"
        self.id_to_obj[rval] = x
        return rval

    def fresh_id(self):
        """Return sequential identifier to guarantee a unique node."""
        self.currid += 1
        return f"Y{self.currid}"

    def _strip_cosmetic(self, node):
        while (
            node
            and node.debug.about
            and node.debug.about.relation == "cosmetic"
        ):
            node = node.debug.about.debug.obj
        return node

    def cynode(self, id, label, classes, parent=None, node=None):
        """Build data structure for a node in cytoscape."""
        if not isinstance(id, str):
            if node is None:
                node = id
            id = self.id(id)
        data = {"id": id, "label": str(label)}
        if self.tooltip_gen and node:
            ttip = self.tooltip_gen(self._strip_cosmetic(node))
            if ttip is not None:
                if not isinstance(ttip, str):
                    ttip = str(hrepr(ttip))
                data["tooltip"] = ttip
        if parent:
            parent = parent if isinstance(parent, str) else self.id(parent)
            data["parent"] = parent
        self.nodes.append({"data": data, "classes": classes})

    def cyedge(self, src_id, dest_id, label):
        """Build data structure for an edge in cytoscape."""
        cl = "input-edge"
        if isinstance(label, tuple):
            label, cl = label
        if not isinstance(label, str):
            label = str(label)
        if not isinstance(dest_id, str):
            dest_id = self.id(dest_id)
        if not isinstance(src_id, str):
            src_id = self.id(src_id)
        data = {
            "id": f"{dest_id}-{src_id}-{label}",
            "label": label,
            "source": dest_id,
            "target": src_id,
        }
        self.edges.append({"data": data, "classes": cl})

    def __hrepr__(self, H, hrepr):
        return H.div["myia-expanded-graph-container"](
            hrepr(
                CytoscapeGraph({
                    **self.cyoptions,
                    "on_node": self.on_node,
                    "style": gcss + self.extra_style,
                    "elements": self.nodes + self.edges,
                    "width": hrepr.config.graph_width or "100%",
                    "height": hrepr.config.graph_height or "500px",
                })
            )["myia-expanded-graph"]
        )


def _has_error(dbg):
    # Whether an error occurred somewhere in a DebugInfo
    if getattr(dbg, "errors", None):
        return True
    elif dbg.about:
        return _has_error(dbg.about)
    else:
        return False


def _make_class_gen(cgen):
    if isinstance(cgen, (tuple, list, set, frozenset)):
        cgen = frozenset(cgen)
        return lambda x, cl: f"error {cl}" if x in cgen else cl
    elif isinstance(cgen, dict):
        return lambda x, cl: f"{cgen[x]} {cl}" if x in cgen else cl
    else:
        return cgen


cosmetics = []


class MyiaGraphPrinter(GraphPrinter):
    """
    Utility to generate a graphical representation for a graph.

    Attributes:
        duplicate_constants: Whether to create a separate node for
            every instance of the same constant.
        duplicate_free_variables: Whether to create a separate node
            to represent the use of a free variable, or point directly
            to that node in a different graph.
        function_in_node: Whether to print, when possible, the name
            of a node's operation directly in the node's label instead
            of creating a node for the operation and drawing an edge
            to it.
        follow_references: Whether to also print graphs that are
            called by this graph.

    """

    def __init__(
        self,
        entry_points,
        *,
        duplicate_constants=False,
        duplicate_free_variables=False,
        function_in_node=False,
        follow_references=False,
        tooltip_gen=None,
        class_gen=None,
        extra_style=None,
        beautify=True,
        on_node=None,
    ):
        """Initialize a MyiaGraphPrinter."""
        super().__init__(
            {"layout": {"name": "dagre", "rankDir": "TB"}},
            tooltip_gen=tooltip_gen,
            extra_style=extra_style,
            on_node=on_node,
        )
        # Graphs left to process
        if beautify:
            self.graphs = set()
            self.focus = set()
            for g in entry_points:
                self._import_graph(g)
        else:
            self.graphs = set(entry_points)
            self.focus = set(self.graphs)

        self.beautify = beautify
        self.duplicate_constants = duplicate_constants
        self.duplicate_free_variables = duplicate_free_variables
        self.function_in_node = function_in_node
        self.follow_references = follow_references
        self.labeler = NodeLabeler(
            function_in_node=function_in_node,
            relation_symbols=short_relation_symbols,
        )
        self._class_gen = _make_class_gen(class_gen)
        # Nodes processed
        self.processed = set()
        # Nodes left to process
        self.pool = set()
        # Nodes that are to be colored as return nodes
        self.returns = set()
        # IDs for duplicated constants
        self.currid = 0

    def _import_graph(self, graph):
        mng = manage(graph, weak=True)
        graphs = set()
        parents = mng.parents
        g = graph
        while g:
            graphs.add(g)
            g = parents[g]
        clone = GraphCloner(*graphs, total=True, relation="cosmetic")
        self.graphs |= {clone[g] for g in graphs}
        self.focus.add(clone[graph])

    def name(self, x):
        """Return the name of a node."""
        return self.labeler.name(x, force=True)

    def label(self, node, fn_label=None):
        """Return the label to give to a node."""
        return self.labeler.label(node, None, fn_label=fn_label)

    def const_fn(self, node):
        """
        Return name of function, if constant.

        Given an `Apply` node of a constant function, return the
        name of that function, otherwise return None.
        """
        return self.labeler.const_fn(node)

    def add_graph(self, g):
        """Create a node for a graph."""
        if g in self.processed:
            return
        if self.beautify:
            g = cosmetic_transformer(g)
        name = self.name(g)
        argnames = [self.name(p) for p in g.parameters]
        lbl = f'{name}({", ".join(argnames)})'
        classes = ["function", "focus" if g in self.focus else ""]
        self.cynode(id=g, label=lbl, classes=" ".join(classes))
        self.processed.add(g)

    def process_node_generic(self, node, g, cl):
        """Create node and edges for a node."""
        lbl = self.label(node)

        self.cynode(id=node, label=lbl, parent=g, classes=cl)

        fn = node.inputs[0] if node.inputs else None
        if fn and fn.is_constant_graph():
            self.graphs.add(fn.value)

        for inp in node.inputs:
            if inp.is_constant_graph():
                self.cyedge(src_id=g, dest_id=inp.value, label=("", "use-edge"))

        edges = []
        if fn and not (fn.is_constant() and self.function_in_node):
            edges.append((node, "F", fn))

        edges += [
            (node, i + 1, inp) for i, inp in enumerate(node.inputs[1:]) or []
        ]

        self.process_edges(edges)

    def class_gen(self, node, cl=None):
        """Generate the class name for this node."""
        g = node.graph
        if cl is not None:
            pass
        elif node in self.returns:
            cl = "output"
        elif node.is_parameter():
            cl = "input"
            if node not in g.parameters:
                cl += " unlisted"
        elif node.is_constant():
            cl = "constant"
        elif node.is_special():
            cl = f"special-{type(node.special).__name__}"
        else:
            cl = "intermediate"
        if _has_error(node.debug):
            cl += " error"
        if self._class_gen:
            return self._class_gen(self._strip_cosmetic(node), cl)
        else:
            return cl

    def process_node(self, node):
        """Create node and edges for a node."""
        if node in self.processed:
            return

        g = node.graph
        self.follow(node)
        cl = self.class_gen(node)
        if g and g not in self.processed:
            self.add_graph(g)

        if node.inputs and node.inputs[0].is_constant():
            fn = node.inputs[0].value
            if fn in cosmetics:
                cosmetics[fn](self, node, g, cl)
            elif hasattr(fn, "graph_display"):
                fn.graph_display(self, node, g, cl)
            else:
                self.process_node_generic(node, g, cl)
        else:
            self.process_node_generic(node, g, cl)

        self.processed.add(node)

    def process_edges(self, edges):
        """Create edges."""
        for edge in edges:
            src, lbl, dest = edge
            if dest.is_constant() and self.duplicate_constants:
                self.follow(dest)
                cid = self.fresh_id()
                self.cynode(
                    id=cid,
                    parent=src.graph,
                    label=self.label(dest),
                    classes=self.class_gen(dest, "constant"),
                    node=dest,
                )
                self.cyedge(src_id=src, dest_id=cid, label=lbl)
            elif (
                self.duplicate_free_variables
                and src.graph
                and dest.graph
                and src.graph is not dest.graph
            ):
                self.pool.add(dest)
                cid = self.fresh_id()
                self.cynode(
                    id=cid,
                    parent=src.graph,
                    label=self.labeler.label(dest, force=True),
                    classes=self.class_gen(dest, "freevar"),
                    node=dest,
                )
                self.cyedge(src_id=src, dest_id=cid, label=lbl)
                self.cyedge(src_id=cid, dest_id=dest, label=(lbl, "link-edge"))
                self.cyedge(
                    src_id=src.graph,
                    dest_id=dest.graph,
                    label=("", "nest-edge"),
                )
            else:
                self.pool.add(dest)
                self.cyedge(src_id=src, dest_id=dest, label=lbl)

    def process_graph(self, g):
        """Process a graph."""
        self.add_graph(g)
        for inp in g.parameters:
            self.process_node(inp)

        if not g.return_:
            return

        ret = g.return_.inputs[1]
        if not ret.is_apply() or ret.graph is not g:
            ret = g.return_

        self.returns.add(ret)
        self.pool.add(ret)

        while self.pool:
            node = self.pool.pop()
            self.process_node(node)

    def process(self):
        """Process all graphs in entry_points."""
        if self.nodes or self.edges:
            return
        while self.graphs:
            g = self.graphs.pop()
            self.process_graph(g)
        return self.nodes, self.edges

    def follow(self, node):
        """Add this node's graph if follow_references is True."""
        if node.is_constant_graph() and self.follow_references:
            self.graphs.add(node.value)

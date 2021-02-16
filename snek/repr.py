import os
from hrepr import hrepr, Hrepr
from myia import abstract as A
from myia import lib
from myia import xtype
from myia.debug.label import short_labeler
from myia.operations import primitives as primops

from .graph import MyiaGraphPrinter


mcss_path = f"{os.path.dirname(__file__)}/assets/myia.css"
mcss = open(mcss_path).read()


def _starname(obj):
    if tag := getattr(obj, "tag", None):
        tag = tag.__qualname__
        return f"★{tag}"
    else:
        typ = obj if isinstance(obj, type) else type(obj)
        return typ.__qualname__.replace("Abstract", "★")


class MyiaHrepr(Hrepr):

    ###########
    # General #
    ###########

    def hrepr_resources(self, t: (lib.ANFNode, A.AbstractValue)):
        return [self.H.style(mcss)]

    ########
    # Misc #
    ########

    def hrepr(self, xs: lib.OrderedSet):
        return self.H.bracketed(
            *map(self, xs),
            start="{",
            end="}",
        )

    ############
    # TypeMeta #
    ############

    def hrepr_short(self, t: xtype.TypeMeta):
        return self.H.span(str(t))

    ############
    # Abstract #
    ############

    def _tracks(self, tracks, without=[A.TYPE]):
        tracks = [(k, v) for k, v in tracks.items()
                  if v not in {lib.ANYTHING, lib.UNKNOWN}
                  and k not in without]
        return [self.H.pair(k, self(v), delimiter="↦") for k, v in tracks]

    def _fromseq(self, xs):
        if isinstance(xs, (list, tuple, lib.Possibilities)):
            return list(map(self, xs))
        else:
            return [self(xs)]

    def hrepr_short(self, a: A.AbstractValue):
        return self.H.instance(f"...", type=_starname(a), short=True)

    def hrepr(self, a: A.AbstractValue):
        return self.H.instance(
            *self._tracks(a.values, without=[]),
            vertical=True,
            type=_starname(a),
        )

    def hrepr(self, a: A.AbstractTuple):
        return self.H.instance(
            *self._fromseq(a.elements),
            *self._tracks(a.values),
            vertical=True,
            type="★Tuple",
        )

    def hrepr(self, a: A.AbstractUnion):
        return self.H.instance(
            *self._fromseq(a.options),
            *self._tracks(a.values),
            vertical=True,
            type="★Union",
        )

    def hrepr(self, a: A.AbstractFunction):
        return self.H.instance(
            *self._fromseq(self.values[A.VALUE]),
            type="★Function",
        )

    def hrepr(self, a: A.AbstractWrapper):
        return self.H.instance(
            self(a.element),
            *self._tracks(a.values),
            type=_starname(a),
        )

    def hrepr(self, a: A.AbstractKeywordArgument):
        return self.H.instance(
            self.H.pair("key", a.key, delimiter="="),
            self.H.pair("argument", self(a.argument), delimiter="="),
            *self._tracks(a.values),
            type=_starname(a),
        )

    def hrepr(self, a: A.AbstractClassBase):
        return self.H.instance(
            [
                self.H.pair(k, self(v), delimiter="::")
                for k, v in a.attributes.items()
            ],
            *self._tracks(a.values),
            vertical=True,
            type=_starname(a),
        )

    #########
    # Nodes #
    #########

    def hrepr_short(self, node: lib.ANFNode):
        class_name = node.__class__.__name__.lower()
        label = short_labeler.label(node, True)
        return self.H.span["node", f"node-{class_name}"](label)

    def hrepr_short(self, node: lib.Apply):
        if (
            len(node.inputs) == 2
            and isinstance(node.inputs[0], lib.Constant)
            and node.inputs[0].value is primops.return_
        ):
            return self(node.inputs[1])["node-return"]
        else:
            return self.hrepr_short[lib.ANFNode](node)

    ##########
    # Graphs #
    ##########

    def hrepr_short(self, g: lib.Graph):
        label = short_labeler.label(g, True)
        return self.H.span["node", f"node-Graph"](label)

    def hrepr(self, g: lib.Graph):
        if self.state.depth > 0 and not self.config.expand_graphs:
            # Fallback to hrepr_short
            return NotImplemented

        dc = self.config.duplicate_constants
        dfv = self.config.duplicate_free_variables
        fin = self.config.function_in_node
        fr = self.config.follow_references
        tgen = self.config.node_tooltip
        cgen = self.config.node_class
        xsty = self.config.graph_style
        beau = self.config.graph_beautify
        on_node = self.config.on_node

        gpr = MyiaGraphPrinter(
            {g},
            duplicate_constants=True if dc is None else dc,
            duplicate_free_variables=True if dfv is None else dfv,
            function_in_node=True if fin is None else fin,
            follow_references=True if fr is None else fr,
            tooltip_gen=tgen,
            class_gen=cgen,
            extra_style=xsty,
            beautify=True if beau is None else beau,
            on_node=on_node,
        )

        gpr.process()
        return self(gpr)

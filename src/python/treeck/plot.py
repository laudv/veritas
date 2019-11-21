import graphviz as gv

class TreePlot:

    def __init__(self, fontsize=10):
        self.g = gv.Graph(format="svg")
        self.g.attr(fontsize=str(fontsize), fontname="monospace")
        self.g.attr("node", fontsize=str(fontsize), fontname="monospace")
        self.index = 0

    def name(self, node):
        return "{}x{}".format(self.index, node.id())

    def add_tree(self, tree):
        g = gv.Graph()
        self.index += 1
        stack = [tree.root()]
        while len(stack) > 0:
            node = stack.pop()
            if node.is_leaf():
                g.node(self.name(node), "{:.3f}".format(node.leaf_value()))
            else:
                g.node(self.name(node), str(node.get_split()))
                stack.append(node.right())
                stack.append(node.left())

            if not node.is_root():
                g.edge(self.name(node.parent()), self.name(node))
        self.g.subgraph(g)

    def add_tree_cmp(self, btree, stree):
        self.index += 1
        stack = [(btree.root(), stree.root())] # (big node, small node)
        while len(stack) > 0:
            bnode, snode = stack.pop()
            if bnode == snode:
                if bnode.is_leaf():
                    self.g.node(self.name(bnode), "{:.3f}".format(bnode.leaf_value()),
                            style="bold", color="darkgreen")
                else:
                    self.g.node(self.name(bnode), "{}".format(bnode.get_split()),
                            style="bold", color="darkgreen")
                    stack.append((bnode.right(), snode.right()))
                    stack.append((bnode.left(), snode.left()))

                if not bnode.is_root():
                    self.g.edge(self.name(bnode.parent()), self.name(bnode),
                            style="bold", color="darkgreen")
            else:
                if bnode.is_leaf():
                    self.g.node(self.name(bnode), "{:.3f}".format(bnode.leaf_value()),
                            color="gray", fontcolor="gray")
                else:
                    self.g.node(self.name(bnode), str(bnode.get_split()),
                            color="gray", fontcolor="gray")
                    stack.append((bnode.right(), snode))
                    stack.append((bnode.left(), snode))

                if not bnode.is_root():
                    self.g.edge(self.name(bnode.parent()), self.name(bnode), color="gray")

    def add_domains(self, domains):
        text = ""
        for i, dom in enumerate(domains):
            if dom.is_everything(): continue
            text += "X{}: {}\\l".format(i, dom)
        self.g.attr(label=text)

    def render(self, f):
        self.g.render(f)

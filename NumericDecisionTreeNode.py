

class NumericDecisionTreeNode(object):

    def __init__(self, parent, children=None, threshold=None, group=None):
        self.parent = parent
        self.group = group
        self.threshold = threshold
        self.children = children
        self.isLeaf = children == None

    def is_root(self):
        return self.parent == None and self.threshold is not None

    def is_leaf(self):
        return self.children is None and self.threshold is None

    def set_child(self, index, node):
        if not self.is_leaf():
            self.children[index] = node

    def __str__(self):
        if self.isLeaf: 
            return "Numeric Decision Conditional Leaf. Group={0}".format(self.group)
        else:
            return "Numeric Decision Conditional Node. Left if x[{0}] <= {1}".format(self.group, self.threshold)


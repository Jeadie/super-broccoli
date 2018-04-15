

class NumericDecisionTreeNode(object):
    """
    An individual node in the DecisionTree path. Is either a leaf node or internal.
    Internal nodes are conditionals where:
        left child satisfies x[group] <= threshold &
        right child x[group] > threshold

    Leaf nodes are final group indexes.
    """

    def __init__(self, parent, children=None, threshold=None, group=None):
        """
        Constructor.
        :param parent: reference to NumericDecisionTreeNode
        :param children: Either None for leaf nodes or an array of references to children for internal nodes
        :param threshold: Either None for leaf nodes or the real threshold value for the internal nodes conditional
        :param group: Either the final resultant group of a leaf node, or the feature index that is the determinant
                        in the conditional
        """
        self.parent = parent
        self.group = group
        self.threshold = threshold
        self.children = children

    def is_root(self):
        """
        :return: True if the node has no parent.
        """
        return self.parent is None

    def is_leaf(self):
        """
        :return: True if the node has no children and is hence a leaf
        """
        return self.children is None

    def set_child(self, index, node):
        """
        Sets the left/right child of this node to the parametered node.
        :param index: 0 for the left child, 1 for the right child
        :param node: the node to set as the child
        """
        if not self.is_leaf():
            self.children[index] = node

    def __str__(self):
        if self.is_leaf():
            return "Numeric Decision Conditional Leaf. Group={0}".format(self.group)
        else:
            return "Numeric Decision Conditional Node. Left if x[{0}] <= {1}".format(self.group, self.threshold)



class NumericDecisionTreeGraph(object):
    def __init__(self, root):
        self.root = root

    def inorder_traversal(self, func):
        """
        Performs an inorder traversal of the graph, applying func at each node.
        :param func: Function that takes in a NumericDecisionTreeNode with the return value ignored.
        """
        self._inorder_recurse(self.root, func)

    def _inorder_recurse(self, node, func):
        func(node)
        if not node.is_leaf():
            self._inorder_recurse(node.children[0], func)
            self._inorder_recurse(node.children[1], func)


    def postorder_traversal(self, func):
        self._postorder_recurse(self.root, func)

    def _postorder_recurse(self, node, func):
        if not node.is_leaf():
            self._inorder_recurse(node.children[0], func)
            self._inorder_recurse(node.children[1], func)
        func(node)


    def preorder_traversal(self, func):
        self._preorder_recurse(self.root, func)

    def _preorder_recurse(self, node, func):
        if not node.is_leaf():
            self._inorder_recurse(node.children[0], func)
            func(node)
            self._inorder_recurse(node.children[1], func)
        else:
            func(node)
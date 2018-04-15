from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from NumericDecisionTreeGraph import NumericDecisionTreeGraph
from NumericDecisionTreeNode import NumericDecisionTreeNode


class NumericDecisionTreeHandler(object):

    def __init__(self):
        self.decisionTree = None

    def construct_decision_tree(self, data, outputs):
        if self.decisionTree is None:
            self.decisionTree = DecisionTreeClassifier()
        self.decisionTree.fit(data, outputs)

    def get_decision_flow(self):
        """
        Returns a NumericDecisionTreeGraph from the given decision tree handled.
        """
        self.initial_tree = self.decisionTree.tree_
        # lists are In order traversals of the tree.
        self.thresholds = self.initial_tree.threshold
        self.features = self.initial_tree.feature
        self.leafs = self.initial_tree.children_left == self.initial_tree.children_right
        self.datapointsAtNode = self.initial_tree.value
        root = NumericDecisionTreeNode(None, children=[None, None], threshold=self.thresholds[0], group=self.features[0])
        left_index = self.initial_tree.children_left[0]
        right_index = self.initial_tree.children_right[0]

        root.set_child(0, self.recursive_construct_child(root, left_index))
        root.set_child(1, self.recursive_construct_child(root, right_index))
        return NumericDecisionTreeGraph(root)

    def recursive_construct_child(self, parent, node_index):
        nodeIsLeaf = self.leafs[node_index]
        if nodeIsLeaf:
            samples = self.datapointsAtNode[node_index]
            return NumericDecisionTreeNode(parent, children=None, threshold=None, group=samples.argmax())

        else:
        # Else, recursively add children
            node = NumericDecisionTreeNode(parent, children=[None, None], threshold=self.thresholds[node_index], group=self.features[node_index])
            left_index = self.initial_tree.children_left[node_index]
            right_index = self.initial_tree.children_right[node_index]
            node.set_child(0, self.recursive_construct_child(node, left_index))
            node.set_child(1, self.recursive_construct_child(node, right_index))
            return node

if __name__ == '__main__':
    iris = load_iris()
    handler = NumericDecisionTreeHandler()
    handler.construct_decision_tree(iris.data, iris.target)
    graph = handler.get_decision_flow()
    graph.inorder_traversal(lambda node: print(node))
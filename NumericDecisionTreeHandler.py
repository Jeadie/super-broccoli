from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from NumericDecisionTreeGraph import NumericDecisionTreeGraph
from NumericDecisionTreeNode import NumericDecisionTreeNode


class NumericDecisionTreeHandler(object):
    """
    Wraps the scikit-learn decision tree classifier logic and constructs a graph algorithm for further processing.
    Classifying only currently implemented for scalar output and strictly real features. As per decision trees,
     missing features in datapoints cannot be processed.
    """

    def __init__(self):
        """
        Constructor.
        """
        self.decisionTree = None

    def construct_decision_tree(self, data, outputs):
        """
        Constructs the classifying decision tree for the numeric data and outputs provided.
        :param data: Is sample x feature sized numpy array with strictly real, numeric values.
        :param outputs: real, numeric output vector with each row of data having a corresponding output.
        """
        self.features = data.shape[1]
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
        self.leafs = self.initial_tree.children_left == -1 # leaf nodes dont have children indexes.
        self.datapointsAtNode = self.initial_tree.value # the number of each category that decisions path cross the indexed node


        root = NumericDecisionTreeNode(None, children=[None, None], threshold=self.thresholds[0], group=self.features[0])
        left_index = self.initial_tree.children_left[0]
        right_index = self.initial_tree.children_right[0]

        # Recursively add two subtrees of root.
        root.set_child(0, self.recursive_construct_child(root, left_index))
        root.set_child(1, self.recursive_construct_child(root, right_index))
        return NumericDecisionTreeGraph(root)

    def recursive_construct_child(self, parent, node_index):
        """
        Recursively constructs the node at a given index from the inorder list and adds its children.
        :param parent: NumericDecisionTreeNode parent of node.
        :param node_index: Index of node to construct and return.
        :return: A constructed child branch of the parent node.
        """
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

# if __name__ == '__main__':
#     iris = load_iris()
#     handler = NumericDecisionTreeHandler()
#     handler.construct_decision_tree(iris.data, iris.target)
#     graph = handler.get_decision_flow()
#     graph.inorder_traversal(lambda node: print(node))
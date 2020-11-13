import copy

# Link between number and node type
NodeTypes = {0: "ROOT", 1: "INTERMEDIATE", 2: "LEAF"}


def getDataSplitByFeatures(feature, datas):
    # d1 corresponds to data which respect the feature selection
    d1 = datas[datas[feature.attr].isin(feature.possibilities)]
    # d2 corresponds to data which doesn't respect the feature selection
    d2 = datas[~datas[feature.attr].isin(feature.possibilities)]
    return d1, d2


def getFeatureGini(feature, datas):
    d1, d2 = getDataSplitByFeatures(feature, datas)
    return d1["Survived"].count() / datas["Survived"].count() * getGini(d1) + d2["Survived"].count() / datas["Survived"].count() * getGini(d2)


def getGini(datas):
    return 1 - sum((datas["Survived"].value_counts() / datas["Survived"].count())**2)


def getMajorityClass(datas):
    return datas["Survived"].value_counts().idxmax()

# The NodeFeature class represents the feature selection which is apply by a Node, e.g Pclass < 3.
# Attributes :
#   - attr : the attribute on which the feature selection is apply (e.g Pclass)
#   - possibilities : the values that can have attr (e.g [1,2])


class NodeFeature:
    def __init__(self, attribute, possibilities):
        self.attr = attribute
        self.possibilities = possibilities

    def __str__(self):
        return self.attr + " " + " ".join([str(v) for v in self.possibilities])

    def __eq__(self, other):
        if not isinstance(other, NodeFeature):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.attr == other.attr and self.possibilities == other.possibilities

# The Node class represents a node on the Tree with some "transparent" attributes.
# Attributes :
#   - NodeType : the type of the node (0 for ROOT, 1 for INTERMEDIARY or 2 for LEAF)
#   - NodeDatas : the datas which are "contains" in the node.
#   - NodeLevel : the level of the node.
#   - NodeFeature : the feature selection of the node (if not LEAF).
#   - Parent : the node's parent (if not ROOT).
#   - NodeAttributes : the attributes on which we can again make selection.
#   - ChildrenNodes : the nodes of which it is the parent.
#   - NodeGini : the gini value associated to this node.


class Node:
    def __init__(self, datas, level, parent=None, yesNode=True):
        self.NodeType = 1
        self.NodeDatas = datas
        self.NodeLevel = level
        self.NodeFeature = -1
        # If the node is not the root node, update attributes according to parent node selection (useless to select again the same attribute and possibilities because the datas on the node are already selected by this attribute and possibilities if parent node split datas with sex 0, useless to split data again with sex 0 because it's already the case, it will cause an infinite loop).
        if parent != None:
            self.NodeAttributes = parent.NodeAttributes.copy()
            if yesNode:
                self.NodeAttributes[parent.NodeFeature.attr] = [
                    x for x in self.NodeAttributes[parent.NodeFeature.attr] if x in parent.NodeFeature.possibilities]
            else:
                self.NodeAttributes[parent.NodeFeature.attr] = [
                    x for x in self.NodeAttributes[parent.NodeFeature.attr] if x not in parent.NodeFeature.possibilities]
        self.NodeGini = getGini(datas)
        self.ChildrenNodes = {}
        self.Parent = parent

        if level == 0:
            self.NodeType = 0

    def SetAttributes(self, attributes):
        self.NodeAttributes = attributes

    def SetLeaf(self, final_class):
        self.NodeType = 2
        self.final_class = final_class

    def SetFeature(self, feature):
        self.NodeFeature = feature

    def AddChildrenNode(self, node, yes):
        if yes:
            self.ChildrenNodes["YES"] = node  # feature selection verified
        else:
            self.ChildrenNodes["NO"] = node

    def CheckData(self, trainingdata):
        return trainingdata[self.NodeFeature.attr] in self.NodeFeature.possibilities


# The DecisionTree class represents the tree itself.
# Attributes :
#   - Layers : the global structure of the tree (dictionnary : key = level, values = node)
#   - complexity : the complexity of the tree (number of leaves).
class DecisionTree:
    def __init__(self):
        self.Layers = {}
        self.complexity = 0

    def AddNode(self, node):
        if node.NodeLevel not in self.Layers:
            self.Layers[node.NodeLevel] = []
        self.Layers[node.NodeLevel].append(node)

    def IncreaseComplexity(self):
        self.complexity += 1

    # Function which classify a row (trainingdata) of the dataset. Try to find a leaf associated to this row.
    def ClassifyData(self, trainingdata):
        currentNode = self.Layers[0][0]

        while currentNode.NodeType != 2:
            if currentNode.CheckData(trainingdata):
                currentNode = currentNode.ChildrenNodes["YES"]
            else:
                currentNode = currentNode.ChildrenNodes["NO"]

        return currentNode.final_class

    def GetMaxLevel(self):
        return len(list(self.Layers.keys()))

    def UpdateComplexity(self):
        self.complexity = 0
        for level in self.Layers:
            for node in self.Layers[level]:
                if node.NodeType == 2:
                    self.complexity += 1

    # Function that aims to clean the tree during pruning, we try to remove all the nodes that have in their parent hierarchy a leaf (the place where we apply prunning).
    def CleanTree(self):
        temp_layers = {}
        for i, level in enumerate(self.Layers):
            for j, node in enumerate(self.Layers[level]):
                to_remove = False
                parentNode = node.Parent
                while parentNode != None:
                    if parentNode.NodeType == 2:
                        to_remove = True
                    parentNode = parentNode.Parent
                if not to_remove:
                    if level not in temp_layers:
                        temp_layers[level] = []
                    temp_layers[level].append(node)

        self.Layers = temp_layers

    def __str__(self):
        text = ""
        for key in self.Layers:
            for node in self.Layers[key]:
                text += NodeTypes[node.NodeType] + "\n"
                text += "Level " + str(node.NodeLevel) + "\n"
                if node.NodeType != 2:
                    text += "Feature " + str(node.NodeFeature) + "\n"
                else:
                    text += "Class " + str(node.final_class) + "\n"
                text += "Gini {} \n".format(round(node.NodeGini, 4))
                if node != self.Layers[key][-1]:
                    text += "*****" + "\n"
            if key != list(self.Layers.keys())[-1]:
                text += " " + "\n"
        return text


def GetBestFeature(CurrentNode):
    min_splitgit = 10
    best_feature = None
    for attr in CurrentNode.NodeAttributes:

        possible_features = []
        for i, v in enumerate(CurrentNode.NodeAttributes[attr]):
            gte, gt, lte, lt = CurrentNode.NodeAttributes[attr][max(i, 0):], CurrentNode.NodeAttributes[attr][max(i+1, 0):], CurrentNode.NodeAttributes[attr][:min(
                i + 1, len(CurrentNode.NodeAttributes[attr]) - 1)], CurrentNode.NodeAttributes[attr][:min(i, len(CurrentNode.NodeAttributes[attr]) - 1)]

            for elm in [gte, gt, lte, lt]:
                # Keep only the features that are useful in our case (avoid also redundancy)
                if elm not in possible_features and len(elm) > 0 and len(elm) != len(CurrentNode.NodeAttributes[attr]) and len(elm) != len(CurrentNode.NodeAttributes[attr]) - 2:
                    possible_features.append(elm)

        for possible_feature in possible_features:
            ftr = NodeFeature(attr, possible_feature)
            gsplit = getFeatureGini(ftr, CurrentNode.NodeDatas)
            # Arbitrary choice to take the feature with the biggest number of possibilities.
            if min_splitgit > gsplit or (min_splitgit == gsplit and len(possible_feature) > len(best_feature.possibilities)):
                min_splitgit = gsplit
                best_feature = ftr
    return best_feature, min_splitgit


def BuildDecisionTree(Datas, Attributes, minNum):
    rootNode = Node(Datas, 0, None)
    # For the root node, we can apply feature selection on base attributes
    rootNode.SetAttributes(Attributes)
    Tree = DecisionTree()
    Build(Datas, minNum, rootNode, Tree)

    return Tree


def Build(Datas, minNum, node, tree):
    tree.AddNode(node)

    if node.NodeGini == 0:  # All datas belongs to the same class.
        node.SetLeaf(Datas["Survived"].iloc[0])
        tree.IncreaseComplexity()
        return

    if Datas["Survived"].count() <= minNum:
        node.SetLeaf(getMajorityClass(node.NodeDatas)) #I deciced to use majority vote to set a class to a node (not default class).
        tree.IncreaseComplexity()
        return

    best_feature, _ = GetBestFeature(node)

    # If we can't do anothoer split on data (all the conditions are already used), we transform the node into a leaf and apply majority voting to the class.
    if best_feature == None:
        node.SetLeaf(getMajorityClass(node.NodeDatas))
        tree.IncreaseComplexity()
        return

    # if node.Parent != None:
    #     print(node.NodeLevel, node.NodeAttributes, " Parent ",
    #                   node.Parent.NodeLevel, node.Parent.NodeAttributes)
    # else:
    #     print(node.NodeLevel, node.NodeAttributes)
    # input()

    node.SetFeature(best_feature)
    d1, d2 = getDataSplitByFeatures(best_feature, node.NodeDatas)
    D = [d1, d2]
    for i, d in enumerate(D):
        # Create new node from feature selection of current node (i == 0 is the condition to know if the node verify parent selection feature : yes edge).
        new_node = Node(d, node.NodeLevel + 1, node, i == 0)
        node.AddChildrenNode(new_node, i == 0)
        # Build recursively the tree
        Build(d, minNum, new_node, tree)


def printDecisionTree(Tree):
    print("Start Tree printing...")
    print("#" * 30)
    print("")
    print(Tree)
    print("#" * 30)


def generalizationError(Datas, Tree, alpha, normalize=False, debug_print=False):
    error = alpha * Tree.complexity
    training_error = 0

    for index, row in Datas.iterrows():
        if Tree.ClassifyData(row) != row['Survived']:
            training_error += 1

    if debug_print:
        print("Number of leaves : " + str(Tree.complexity))
        print("Training error : {}/{}".format(training_error,
                                              Datas["Survived"].count()))
        input()

    error += training_error
    if normalize:
        error = error / Datas["Survived"].count()
    return error


def pruneTree(Tree, alpha, minNum):
    total_datas = Tree.Layers[0][0].NodeDatas
    max_level = Tree.GetMaxLevel()
    for l in range(max_level - 1, -1, -1):
        for i, node in enumerate(Tree.Layers[l]):
            if node.NodeType != 1:  # Check if the node is an intermediary node
                continue

            Subtree = copy.deepcopy(Tree)
            Subtree.Layers[l][i].ChildrenNodes = {}

            # Transform the intermediary node into leaf node.
            # If there is only one class, this line will also select the only class which is represent.
            Subtree.Layers[l][i].SetLeaf(
                getMajorityClass(Subtree.Layers[l][i].NodeDatas))

            Subtree.CleanTree()
            Subtree.UpdateComplexity()

            # If the Subtree is better than the original Tree, switch them (lower error for the Subtree)
            if generalizationError(total_datas, Tree, alpha, True) > generalizationError(total_datas, Subtree, alpha, True):
                Tree = Subtree

    return Tree

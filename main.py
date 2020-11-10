from decision_functions import *
import pandas as pd

datas = pd.read_csv('data/data.csv')
attributes = {}
for key in datas: #Create the "base" list of the attributes that we can use to classify our data and construct our tree.
    if key != "Survived":
        attributes[key] = list(datas[key].unique())
        attributes[key].sort()

errors_string = []

Tree = BuildDecisionTree(datas, attributes, 5)
printDecisionTree(Tree)

with open("output_tree.txt", "w") as output_tree: 
    output_tree.write(str(Tree)) 

errors_string.append("Generalization error before prunning with normalization : {:.4f} \n".format(generalizationError(datas, Tree, 0.5, True)))
print(errors_string[0])

PruneTree = pruneTree(Tree, 0.5, 5)
printDecisionTree(PruneTree)

with open("postpruned_tree.txt", "w") as postpruned_tree: 
    postpruned_tree.write(str(PruneTree)) 

errors_string.append("Generalization error after prunning with normalization : {:.4f}".format(generalizationError(datas, PruneTree, 0.5, True)))
print(errors_string[1])

with open("generalization_error.txt", "w") as error: 
    error.writelines(errors_string)

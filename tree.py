import pandas as pd
import seaborn as sns
sns.set()

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree


def calc_impurity(tree, index):
    # print("index: ", index, " impurity: ", d_tree.tree_.n_node_samples[index] * tree.impurity[index] / 10000000)
    # wenn es 'children' besuche die 'children'
    if tree.children_left[index] != _tree.TREE_LEAF:
        costs_left, leafs_left = calc_impurity(tree, tree.children_left[index])
        costs_right, leafs_right = calc_impurity(tree, tree.children_right[index])

        return costs_left + costs_right, leafs_left + leafs_right
    # wenn es keine 'children' gibt bin ich ein leaf Knoten
    else:
        # print("index: ", index, " cost: ", d_tree.tree_.n_node_samples[index] * tree.impurity[index]/10000000)
        return d_tree.tree_.n_node_samples[index] * tree.impurity[index]/10000000, 1

def determin_alpha(tree):
    min_gk = 1000000
    min_node_idx = tree.node_count
    for node_idx in range(d_tree.tree_.node_count):
        if tree.children_left[node_idx] == _tree.TREE_LEAF:
            continue

        # inner node
        subtree_impurity, subtree_leafs = calc_impurity(tree, node_idx)
        own_impurity = tree.n_node_samples[node_idx] * tree.impurity[node_idx] / 10000000
        # print(node_idx)
        # print("leafs", subtree_leafs)
        gk = (own_impurity - subtree_impurity) / (subtree_leafs - 1.)
        # print("gk: ", gk)
        if gk < min_gk:
            min_node_idx = node_idx
            min_gk = gk

    return min_node_idx, min_gk

if __name__ == '__main__':
    data = pd.read_csv("data/housing/housing.csv")

    input_features = data[['latitude', 'longitude']]
    target = data['median_house_value']

    d_tree = DecisionTreeRegressor(max_depth=10)
    d_tree.fit(input_features, target)

    d_tree.predict(input_features)

    alpha = 0
    k = 1

    determin_alpha
    min_node_idx, min_gk = determin_alpha(d_tree.tree_)

    print("alpha_new: ", min_gk)
    print("alpha_new_node_idx: ", min_node_idx)

    # print(d_tree.tree_.children_left)
    # print(d_tree.tree_.node_count)
    # print(d_tree.tree_.n_node_samples)



def prune(inner_tree, index, threshold):
    # wenn es 'children' besuche die 'children'
    if inner_tree.children_left[index] != _tree.TREE_LEAF:
        prune(inner_tree, inner_tree.children_left[index], threshold)
        prune(inner_tree, inner_tree.children_right[index], threshold)

    # wenn es keine 'children' gibt bin ich ein leaf knoten
    else:
        print(index)



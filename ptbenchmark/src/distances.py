from functools import lru_cache
import numpy as np

def get_leaves(tree):
    leaves = []
    for i in range(tree.N):
        children = tree.get_children(i)
        if len(children) == 0:
            leaves.append(i)
    return set(leaves)


def compute_subtree_leaves(tree):
    @lru_cache(maxsize=None)
    def dfs(node):
        children = tree.get_children(node)
        if len(children) == 0:
            return {node}  # Leaf node.
        leaf_set = set()
        for c in children:
            leaf_set |= dfs(c)
        return leaf_set

    # Compute the leaf set for every node.
    subtree_leaves = {}
    for i in range(tree.N):
        subtree_leaves[i] = dfs(i)
    return subtree_leaves

def get_splits(tree):
    total_leaves = get_leaves(tree)
    subtree_leaves = compute_subtree_leaves(tree)
    splits = set()

    for node, leaves in subtree_leaves.items():
        # A split is valid when:
        # - the node is not the root (parent != -1);
        # - it contains at least two leaves;
        # - it does not contain every leaf.
        if tree.get_parent(node) != -1 and 1 < len(leaves) < len(total_leaves):
            splits.add(frozenset(leaves))

    return splits, len(total_leaves)


def robinson_foulds_distance(tree1, tree2, normalize=True):
    splits1, nleaves1 = get_splits(tree1)
    splits2, nleaves2 = get_splits(tree2)

    only_in_1 = splits1 - splits2
    only_in_2 = splits2 - splits1
    rf = len(only_in_1) + len(only_in_2)
    rows = tree1.rows
    cols = tree1.cols
    
    if normalize:
        return rf / (rows*cols/2)
    else:
        return rf

def root_distance(tree1, tree2, normalize=True):
    tree1_dist = tree1.dist_from_root()
    tree2_dist = tree2.dist_from_root()
    if normalize:
        return np.sum((tree1_dist - tree2_dist)**2) / tree2_dist.size / max(np.max(tree1_dist), np.max(tree1_dist))
    else:
        return np.sum((tree1_dist - tree2_dist)**2)

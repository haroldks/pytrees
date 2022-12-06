import copy
import json
import random
from typing import List


def get_random_feature(nodes: List):
    if len(nodes) > 0:
        node = nodes.pop(random.randint(0, len(nodes) - 1))
        return node


def gen_node(feature, depth):
    return {
        "root": feature,
        "left": None,
        "right": None,
        "depth": depth,
        "class": None,
        "is_leaf": False,
    }


def prune(tree, v=0):
    if "meta" in tree.keys() and tree["meta"] == -1 and not tree["is_leaf"]:
        tree["left"] = None
        tree["right"] = None
        tree["is_leaf"] = True
        tree["class"] = v
        tree["root"] = None
    else:
        if not tree["is_leaf"]:
            prune(tree["left"], v=0)
            prune(tree["right"], v=1)
    return tree


def generate_decision_trees(
    depth,
    to_prune=False,
    save=None,
    both=False,
    name="{folder}/{name}_{type}_d_{depth}.json",
):
    def recursion(tree, nodes, remaining_depth, val=None):
        if remaining_depth == 0:
            root = tree["root"]
            tree["class"] = val
            nodes.append(root)
            tree["root"] = None
            tree["is_leaf"] = True
        else:
            # Left
            val = None
            meta = random.choices([-1, 1], weights=[0.49, 0.51])[0]
            tree["left"] = gen_node(
                get_random_feature(nodes), depth - remaining_depth + 1
            )
            left_class = random.choice([0, 1])

            tree["left"]["meta"] = meta

            if remaining_depth == 1:
                val = left_class

            recursion(tree["left"], nodes, remaining_depth - 1, val=val)

            # Right
            meta = random.choices([-1, 1], weights=[0.49, 0.51])[0]
            if remaining_depth == 1:
                val = (val + 1) % 2
            tree["right"] = gen_node(
                get_random_feature(nodes), depth - remaining_depth + 1
            )
            tree["right"]["meta"] = meta
            recursion(tree["right"], nodes, remaining_depth - 1, val=val)

        return tree

    max_nodes = 2**depth - 1
    nodes = random.sample(range(max_nodes), max_nodes)
    feature = get_random_feature(nodes)
    main_tree = gen_node(feature, 0)
    main_tree = recursion(main_tree, nodes, depth)
    pruned_tree = None
    if to_prune:
        pruned_tree = prune(copy.deepcopy(main_tree))

    if save:
        if to_prune:
            with open(name.format(type="pruned", depth=depth), "w") as f:
                json.dump(pruned_tree, f)
            if both:
                with open(name.format(type="full", depth=depth), "w") as f:
                    json.dump(main_tree, f)
        else:
            with open(name.format(type="full", depth=depth), "w") as f:
                json.dump(main_tree, f)

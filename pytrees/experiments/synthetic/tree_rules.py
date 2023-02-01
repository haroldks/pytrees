import copy
import json


def get_tree_rules(tree, rule=None):
    if rule is None:
        rule = {}
    rules = list()
    if tree["is_leaf"]:
        rule["target"] = tree["class"]
        return rule
    else:

        # Move to the left ?
        root = tree["root"]

        sub_tree = tree["left"]

        rule_copy = copy.deepcopy(rule)

        rule[root] = 0
        new_rules = get_tree_rules(sub_tree, rule=rule)
        if isinstance(new_rules, list):
            rules = rules + new_rules
        else:
            rules.append(new_rules)

        # Move to the right ?

        sub_tree = tree["right"]
        rule_copy[root] = 1
        new_rules = get_tree_rules(sub_tree, rule=rule_copy)
        if isinstance(new_rules, list):
            rules = rules + new_rules
        else:
            rules.append(new_rules)

        return rules


def get_rules_from_decision_tree(tree_path):
    tree_path = tree_path + ".json" if not tree_path.endswith(".json") else tree_path

    with open(tree_path, "r") as file:
        tree = json.load(file)

    rules = get_tree_rules(tree, rule={})
    rules = sorted(rules, key=lambda r: r["target"])

    return rules

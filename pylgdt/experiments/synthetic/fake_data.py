import random
import numpy as np
from .tree_rules import get_rules_from_decision_tree


def generate_dataset_from_decision_tree(
    tree_path, n_attributes=0, samples=100, save=None
):
    rules = get_rules_from_decision_tree(tree_path)

    rules_distributions = {0: 0, 1: 0}
    m = 0

    for rule in rules:
        # print(rule)
        rules_distributions[rule["target"]] += 1
        mx = max(list(rule.keys())[0:-1])
        m = mx if mx > m else m
    n_attributes = max(m + 1, n_attributes)
    n_attributes = random.randint(n_attributes, n_attributes + 20)
    assert n_attributes > m, f"n_attributes must be greater than {m}"
    samples = max(samples, 2 * max(rules_distributions.values()))
    assert samples >= 2 * max(
        rules_distributions.values()
    ), f"Samples should be greater or equal to {2 * max(rules_distributions.values())} "

    samples = samples + 1 if not samples % 2 == 0 else samples

    rules_samples = {0: [0, 0], 1: [0, 0]}

    for key in rules_samples.keys():
        rules_samples[key] = [
            (samples // 2) // rules_distributions[key],
            (samples // 2) % rules_distributions[key],
        ]
    print(rules_distributions)

    data = list()

    for target in rules_distributions.keys():

        treshold = target * rules_distributions[(target - 1) % 2]

        for i in range(rules_distributions[target]):
            index = i + treshold
            for _ in range(rules_samples[target][0]):
                buff = [random.randint(0, 1) for _ in range(n_attributes)]
                for attribute in list(rules[index].keys())[0:-1]:
                    buff[attribute] = rules[index][
                        attribute
                    ]  # TODO: Here I can add noise in rule with target and attribute to see behaviour

                data.append([target] + buff)

    for target in rules_samples.keys():
        treshold = target * rules_distributions[(target - 1) % 2]
        for i in range(rules_samples[target][1]):
            index = treshold + i
            buff = [random.randint(0, 1) for _ in range(n_attributes)]
            for attribute in list(rules[index].keys())[0:-1]:
                buff[attribute] = rules[index][attribute]
            data.append([target] + buff)
            # print(([target] + buff))
            # print(data)
    data = np.asarray(data)

    if save is not None:
        np.savetxt(save, data, fmt="%d")

    return np.asarray(data)

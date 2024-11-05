import random
from node import Node
from collections import defaultdict
from parse import parse
import math
from ID3 import calculate_information_gain, get_attribute_values, get_examples_with_value

class RandomForest:
    def __init__(self, num_trees=10, max_features=None):
        self.num_trees = num_trees  # Number of trees in the forest
        self.max_features = max_features  # Maximum number of features to consider for splits
        self.trees = []

    def fit(self, examples):
        """
        Trains the random forest by creating multiple decision trees on different subsets of the data.
        """
        n_samples = len(examples)
        for _ in range(self.num_trees):
            # Bootstrap sample: random subset of the data (with replacement)
            bootstrap_sample = [examples[random.randint(0, n_samples - 1)] for _ in range(n_samples)]
            tree = self.build_tree(bootstrap_sample)
            self.trees.append(tree)

    def build_tree(self, examples):
        """
        Builds a decision tree using the ID3 algorithm but with random attribute selection.
        """
        default_class = self.most_common_class(examples)
        return self.ID3(examples, default_class)

    def ID3(self, examples, default):
        """
        Builds a decision tree using a random subset of attributes at each split.
        """
        if not examples:
            return Node(label=default)

        # Check if all examples have the same class
        if self.is_pure(examples):
            return Node(label=examples[0]["Class"])

        # Select random subset of attributes for splitting
        attributes = list(examples[0].keys())
        attributes.remove("Class")

        if self.max_features:
            attributes = random.sample(attributes, self.max_features)
        
        # Find the best attribute to split on
        best_attr = self.best_attribute(examples, attributes)

        if not best_attr:
            return Node(label=default)

        root = Node(key=best_attr)
        attr_values = get_attribute_values(examples, best_attr)

        for value in attr_values:
            sub_examples = get_examples_with_value(examples, best_attr, value)
            if not sub_examples:
                root.children[value] = Node(label=default)
            else:
                subtree = self.ID3(sub_examples, default)
                root.children[value] = subtree

        return root

    def predict(self, example):
        """
        Predict the class label for a single example by majority voting from all decision trees.
        """
        votes = [self.evaluate(tree, example) for tree in self.trees]
        return max(set(votes), key=votes.count)

    def evaluate(self, node, example):
        """
        Evaluates a single example using a decision tree.
        """
        if node.label is not None:
            return node.label

        value = example.get(node.key, "?")
        if value == "?" or value not in node.children:
            return self.most_common_class(node.children.values())

        return self.evaluate(node.children[value], example)

    def most_common_class(self, examples):
        """
        Determines the most common class in a set of examples.
        """
        class_counts = defaultdict(int)
        for example in examples:
            class_counts[example["Class"]] += 1
        return max(class_counts, key=class_counts.get)

    def best_attribute(self, examples, attributes):
        """
        Determines the best attribute for splitting using information gain.
        """
        info_gains = {attr: calculate_information_gain(examples, attr) for attr in attributes}
        return max(info_gains, key=info_gains.get, default=None)

    def is_pure(self, examples):
        """
        Checks if all examples have the same class label.
        """
        first_class = examples[0]["Class"]
        return all(example["Class"] == first_class for example in examples)

data = parse("HW#1/candy.data")
random.shuffle(data)
train = data[:len(data) // 2]
test = data[len(data) // 2 :]
rf = RandomForest(10, 9)
rf.fit(train)

prediction = rf.predict(test)
print("Predicted class:", prediction)
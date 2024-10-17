from node import Node
import math
from collections import defaultdict


def ID3(examples, default):
    """
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    """
    # If examples empty, return leaf node with default class
    if not examples:
        return Node(label=default)

    # Get the majority class from current examples to use as default
    default = get_majority_class(examples)

    # Base case 2: If all examples have same class, return leaf node with that class
    if all_same_class(examples):
        return Node(label=examples[0]["Class"])

    # Get list of all attributes except the class attribute
    attributes = get_attributes(examples)

    # Base case 3: If no attributes left, return leaf node with majority class
    if not attributes:
        return Node(label=default)

    # Find the attribute that gives the highest information gain
    best_attribute = find_best_attribute(examples, attributes)

    # Create a new internal node using the best attribute
    root = Node(key=best_attribute)

    # Get all possible values for the best attribute
    attribute_values = get_attribute_values(examples, best_attribute)

    # For each value of the best attribute, create a subtree
    for value in attribute_values:
        # Get subset of examples that have this value for best_attribute
        examples_i = get_examples_with_value(examples, best_attribute, value)

        # If no examples with this value, create leaf node with default class
        if not examples_i:
            root.children[value] = Node(label=default)
        else:
            # Recursively build subtree for this subset of examples
            subtree = ID3(examples_i, default)
            root.children[value] = subtree
            subtree.parent = root  # Set parent pointer for pruning

    return root


def prune(node, examples):
    """
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    """
    if not node or not examples:
        return

    # If it's already a leaf node, nothing to prune
    if not node.children:
        return

    # First, recursively prune children
    # We need to group examples by their attribute values
    for value, child in node.children.items():
        # Get examples that would reach this child
        child_examples = [e for e in examples if e.get(node.key) == value]
        prune(child, child_examples)

    # After pruning children, check if this node should be pruned
    # Calculate accuracy before pruning
    pre_prune_correct = sum(1 for e in examples if evaluate(node, e) == e["Class"])
    pre_prune_accuracy = pre_prune_correct / len(examples) if examples else 0

    # Store original state
    original_children = node.children.copy()
    original_key = node.key

    # Try converting to leaf node with majority class from examples
    class_counts = defaultdict(int)
    for example in examples:
        class_counts[example["Class"]] += 1

    majority_class = (
        max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else node.label
    )

    # Convert to leaf node temporarily
    node.children = {}
    node.key = None
    node.label = majority_class

    # Calculate accuracy after pruning
    post_prune_correct = sum(1 for e in examples if evaluate(node, e) == e["Class"])
    post_prune_accuracy = post_prune_correct / len(examples) if examples else 0

    # If accuracy decreased or stayed the same, revert the pruning
    if post_prune_accuracy <= pre_prune_accuracy:
        node.children = original_children
        node.key = original_key
        node.label = None


def test(node, examples):
    """
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    """
    if not examples:
        return 0

    # Count correct predictions
    correct = 0
    for example in examples:
        if evaluate(node, example) == example["Class"]:
            correct += 1

    return correct / len(examples)


def evaluate(node, example):
    """
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    """
    # Base case: if this is a leaf node, return its label
    if node.label is not None:
        return node.label

    # Get the value of the attribute we're testing at this node
    value = example.get(node.key, "?")

    # Handle missing values (denoted by '?')
    if value == "?":
        # Use majority class across all children
        majority_counts = defaultdict(int)
        total_weight = 0

        # Weight each child's contribution by number of training examples
        for child in node.children.values():
            if child.label is not None:
                # For leaf nodes, add their class
                majority_counts[child.label] += 1
                total_weight += 1
            else:
                # For internal nodes, recursively evaluate
                subtree_result = evaluate(child, example)
                if subtree_result is not None:
                    majority_counts[subtree_result] += 1
                    total_weight += 1

        # If we found any valid classes, return the majority
        if total_weight > 0:
            return max(majority_counts.items(), key=lambda x: x[1])[0]
        # If no valid classes found, propagate up the tree
        return None

    # If value not found in children, return majority class from node
    if value not in node.children:
        majority_class = get_majority_class_from_node(node)
        return majority_class if majority_class is not None else node.label

    # Recursively evaluate the appropriate child node
    result = evaluate(node.children[value], example)
    return result if result is not None else node.label


# Helper functions
def get_majority_class_from_node(node):
    """
    Determines most common class from a node's children.
    Args:
        node: The node to analyze
    Returns:
        The most common class among child nodes or None if no majority found
    """
    class_counts = {}

    # Count classes from all children
    for child in node.children.values():
        if child.label is not None:
            if child.label not in class_counts:
                class_counts[child.label] = 0
            class_counts[child.label] += 1
        else:
            # For internal nodes, recursively get majority class
            child_majority = get_majority_class_from_node(child)
            if child_majority is not None:
                if child_majority not in class_counts:
                    class_counts[child_majority] = 0
                class_counts[child_majority] += 1

    # Return the majority class if we found any
    if not class_counts:
        return None

    return max(class_counts, key=class_counts.get)


# Helper Functions


def get_attributes(examples):
    """
    Returns list of all attributes except the class attribute.
    """
    if not examples:
        return []
    return [attr for attr in examples[0].keys() if attr != "Class"]


def get_attribute_values(examples, attribute):
    """
    Gets all unique values for a given attribute, excluding missing values.
    Args:
        examples: List of example dictionaries
        attribute: The attribute to get values for
    Returns:
        set: All unique values for the attribute
    """
    values = set()
    for example in examples:
        if example[attribute] != "?":
            values.add(example[attribute])
    return values


def all_same_class(examples):
    """
    Checks if all examples have the same class value.
    Args:
        examples: List of example dictionaries
    Returns:
        bool: True if all examples have same class, False otherwise
    """
    if not examples:
        return True
    first_class = examples[0]["Class"]
    return all(ex["Class"] == first_class for ex in examples)


def get_majority_class(examples):
    """
    Determines the most common class in a set of examples.
    Args:
        examples: List of example dictionaries
    Returns:
        The most common class value
    """
    if not examples:
        return None
    class_counts = defaultdict(int)
    for example in examples:
        class_counts[example["Class"]] += 1
    return max(class_counts.items(), key=lambda x: x[1])[0]


def get_examples_with_value(examples, attribute, value):
    """
    Returns subset of examples with given attribute value.
    Args:
        examples: List of example dictionaries
        attribute: The attribute to filter on
        value: The value to filter for
    Returns:
        list: Filtered examples
    """
    return [ex for ex in examples if ex[attribute] == value]


def calculate_entropy(examples):
    """
    Calculates entropy for a set of examples.
    Entropy = -Σ(p_i * log2(p_i)) where p_i is probability of class i
    Args:
        examples: List of example dictionaries
    Returns:
        float: Entropy value
    """
    if not examples:
        return 0

    # Count occurrences of each class
    class_counts = defaultdict(int)
    total = len(examples)

    for example in examples:
        class_counts[example["Class"]] += 1

    # Calculate entropy using the entropy formula
    entropy = 0
    for count in class_counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def calculate_information_gain(examples, attribute):
    """
    Calculates information gain for splitting on an attribute.
    Gain(S,A) = Entropy(S) - Σ((|Sv|/|S|) * Entropy(Sv))
    Args:
        examples: List of example dictionaries
        attribute: Attribute to calculate gain for
    Returns:
        float: Information gain value
    """
    # Calculate entropy before split
    initial_entropy = calculate_entropy(examples)

    # Get all possible values for this attribute
    values = get_attribute_values(examples, attribute)

    # Calculate weighted entropy after split
    weighted_entropy = 0
    total_examples = len(examples)

    for value in values:
        examples_i = get_examples_with_value(examples, attribute, value)
        weight = len(examples_i) / total_examples
        weighted_entropy += weight * calculate_entropy(examples_i)

    # Information gain is difference between initial and weighted entropy
    return initial_entropy - weighted_entropy


def find_best_attribute(examples, attributes):
    """
    Finds attribute with highest information gain.
    Args:
        examples: List of example dictionaries
        attributes: List of attributes to consider
    Returns:
        str: Name of the best attribute
    """
    if not attributes:
        return None

    # Calculate information gain for each attribute
    gains = {attr: calculate_information_gain(examples, attr) for attr in attributes}

    # Return attribute with highest gain
    return max(gains.items(), key=lambda x: x[1])[0]

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
    if not examples: #empty examples
        return Node(label=default)

    classes = {}
    for example in examples:
        if example["Class"] in classes:
            classes[example["Class"]] += 1
        else:
            classes[example["Class"]] = 1
    most_freq_class = max(classes.items(), key=lambda x: x[1])[0]#most frequent class

    in_similar_class = True
    initial_class = examples[0]["Class"]
    for example in examples:
        if example["Class"] != initial_class:
            in_similar_class = False
            break
        
    if in_similar_class:
        return Node(label=initial_class) #return leaf node if all have same class

    
    all_attributes = [] #stores all attributes
    for attr in examples[0].keys():
        if attr != "Class":#not calss attribute
            all_attributes.append(attr) #

    if len(all_attributes) == 0:#no attributes
        return Node(label=most_freq_class)

    #find attr with most info gain from all attributes lisr
    info_gains = {}
    for attr in all_attributes:
        info_gain = calculate_information_gain(examples, attr)
        info_gains[attr] = info_gain

    best_attr = max(info_gains, key=info_gains.get)

    root = Node(key=best_attr)# best attr as root node

    attr_values = get_attribute_values(examples, best_attr) #all values of best attribute

    for value in attr_values:#build subtree recursively
        # subset of examples with this value for best_attr
        sub_examples = get_examples_with_value(examples, best_attr, value)

        if not sub_examples: #no examples
            root.children[value] = Node(label=most_freq_class)
        else:#recursively build subtree for this subset of examples
            subtree = ID3(sub_examples, most_freq_class)
            root.children[value] = subtree
            subtree.parent = root  # Set parent pointer for pruning

    return root


def prune(node, examples):
    """
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    """
    #base case, no node or examples per se
    if node is None or len(examples) == 0: 
        return

    if len(node.children) == 0: #leaf node, nothing to prune
        return

    for value in node.children:#prune each child node
        child = node.children[value]

        # Get examples that should go to this child
        child_examples = []
        for example in examples:
            if example.get(node.key) == value:
                child_examples.append(example)
        
        # Recursively prune the child node
        prune(child, child_examples)

    # After pruning the children, check if we should prune the current node
    # Calculate the accuracy before pruning
    before_prune_correct = 0
    for example in examples:
        if evaluate(node, example) == example["Class"]:
            before_prune_correct += 1

    if len(examples) > 0:
        before_prune_accuracy = before_prune_correct / len(examples)
    else:
        before_prune_accuracy = 0

    # Save current state of the node (key and children)
    original_children = node.children.copy()
    original_key = node.key

    # Determine majority class in the examples at this node
    class_counts = defaultdict(int)
    for example in examples:
        class_counts[example["Class"]] += 1

    if len(class_counts) > 0:
        majority_class = max(class_counts.items(), key=lambda x: x[1])[0]
    else:
        majority_class = node.label

    #convert cur node to leaf node with majority class
    node.children = {}
    node.key = None
    node.label = majority_class

    # determine post prune accuracy
    after_prune_correct = 0
    for example in examples:
        if evaluate(node, example) == example["Class"]:
            after_prune_correct += 1

    if len(examples) > 0:
        after_prune_accuracy = after_prune_correct / len(examples)
    else:
        after_prune_accuracy = 0

    # if pruning process reduces accuracy, revert to original state
    if after_prune_accuracy <= before_prune_accuracy:
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
    if node.key in example:
        value = example[node.key]
    else:
        value = "?"

    # Handle missing values (denoted by '?')
    if value == "?":
        # Use majority class across all children
        majority_counts = {}
        total_weight = 0

        # Weight each child's contribution by number of training examples
        for child in node.children.values():
            if child.label is not None:
                # For leaf nodes, add their class
                if child.label not in majority_counts:
                    majority_counts[child.label] = 1
                majority_counts[child.label] += 1
                total_weight += 1
            else:
                # For internal nodes, recursively evaluate
                subtree_result = evaluate(child, example)
                if subtree_result is not None:
                    if subtree_result not in majority_counts:
                        majority_counts[subtree_result] = 1
                    majority_counts[subtree_result] += 1
                    total_weight += 1

        # If we found any valid classes, return the majority
        if total_weight > 0:
            max_label = None
            max_count = -1
            for label, count in majority_counts.items():
                if count > max_count:
                    max_count = count
                    max_label = label
            return max_label
        # If no valid classes found, propagate up the tree
        return None

    # If value not found in children, return majority class from node
    if value not in node.children:
        majority_class = get_majority_class_from_node(node)
        if majority_class is not None:
            return majority_class

    # Recursively evaluate the appropriate child node
    result = evaluate(node.children[value], example)
    if result is not None:
        return result
    return node.label


# Helper functions
def get_majority_class_from_node(node):
    """
    gets most common class from a node's children.
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


def get_attribute_values(examples, attribute):
    """
    Gets all unique values for a given attribute, excluding missing values.
    """
    values = []
    for example in examples:
        if example[attribute] != "?":
            if example[attribute] not in values:
                values.append(example[attribute])
    return values


def get_examples_with_value(examples, attribute, value):
    """
    Returns subset of examples with given attribute value.
    """
    examplelist = []

    for example in examples:
        if example[attribute] == value:
            examplelist.append(example)

    return examplelist


def calculate_entropy(examples):
    """
    Calculates entropy for a set of examples.
    """
    if not examples:
        return 0

    # Count occurrences of each class
    class_counts = {}

    for example in examples:
        if example["Class"] not in class_counts:
            class_counts[example["Class"]] = 0
        class_counts[example["Class"]] += 1

    # Calculate entropy using the entropy formula
    entropy = 0
    total = len(examples)
    for count in class_counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def calculate_information_gain(examples, attribute):
    """
    Calculates information gain for splitting on an attribute.
    """
    # Calculate entropy before split
    initial_entropy = calculate_entropy(examples)

    # Get all possible values for this attribute
    values = get_attribute_values(examples, attribute)

    # Calculate weighted entropy after split
    weighted_entropy = 0
    total_examples = len(examples)

    for value in values:
        examples = get_examples_with_value(examples, attribute, value)
        weight = len(examples) / total_examples
        weighted_entropy += weight * calculate_entropy(examples)

    return initial_entropy - weighted_entropy
class Node:
    def __init__(self, label=None, children=None, parent=None, key=None):
        self.label = label
        self.children = children if children is not None else {}
        self.parent = parent
        self.key = key

    def add_child(self, value, child):
        """
        Adds a child node to this node's children dictionary.
        
        Args:
            value: The attribute value leading to this child
            child: The child Node object
        """
        self.children[value] = child
        child.parent = self

    def set_label(self, label):
        """
        Sets the label for this node.
        
        Args:
            label: The label to set for this node
        """
        self.label = label

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        if self.label is not None:
            return f"Leaf({self.label})"
        return f"Node(key={self.key}, children={len(self.children)})"
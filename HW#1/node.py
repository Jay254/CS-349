class Node:
    def __init__(self, label=None, children=None, parent=None, key=None):
        self.label = label
        self.children = children if children is not None else {}
        self.parent = parent
        self.key = key

    def add_child(self, value, child):
        self.children[value] = child
        child.parent = self

    def set_label(self, label):
        self.label = label

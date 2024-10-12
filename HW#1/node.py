class Node:
  def __init__(self, label = None, children = {}, parent = None, key = None):
    self.label = label
    self.children = children
    self.parent = parent
    self.key = key
	# you may want to add additional fields here...

  def add_children(self, child):
    self.children.append(child)


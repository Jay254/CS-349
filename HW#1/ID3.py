from node import Node
import math

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

  target_poss = {} 

  for i in examples:
    target_poss[i["Class"]] += 1;
  



  

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

def calculate_entropy(dictionary):
  '''
  Helper function to calculate entropy for a chosen attribute.
  '''
  entropy = 0
  totalCount = sum(dictionary.values())
  for label, count in dictionary:
    if count > 0:
      entropy = - (count/totalCount) * math.log2(count/totalCount)
  return entropy

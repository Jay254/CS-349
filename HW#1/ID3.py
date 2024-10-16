from node import Node
import math

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  t = Node()
  target_poss = {} 
  no_attribs = True

  # how many times each Class variable appears
  for i in examples:
    key = i["Class"]
    target_poss[key] += 1;
    if len(target_poss[key]) > 1:
      # if there are attributes in a dataset, then
      # we can compute it
      no_attribs = False

  if no_attribs:
    return t

  entropy = calculate_entropy(target_poss)

  max_keys = dict_max(entropy)
  
  if len(max_keys) > 1:
    # if there is more than one max key
    # just use given default
    t.set_label(default)
  else:
    t.set_label(max_keys[0])

def dict_max(d):
  '''
  Takes a dictionary
  Returns list of keys with maximum values
  '''
  max_val = 0
  max_keys = []

  for i in d.keys:
    key = i
    val = d[i]
    if max_val < val:
      max_val = val
      max_keys = [key]
    elif max_val == val:
      max_keys.append(key)

  return max_keys

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
  Helper function to calculate entropy. Returns a dictionary of attributes with entropy values.
  '''
  #Create empty dictionary
  entropyDictionary = {}
  #Calculate total count for probability calculation
  totalCount = sum(dictionary.values())
  #Loop through dictionary, calculate entropy for each label
  for label, count in dictionary.items():
    
    if count > 0:
      #Calculate probability
      probability = count/totalCount
      #Calculate entropy
      entropy = - (probability) * math.log2(probability)
    #Assign calculated entropy to label
    entropyDictionary[label] = entropy

  return entropyDictionary

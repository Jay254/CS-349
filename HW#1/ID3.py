from node import Node
import math
# TODO: Get rid of 'import parse' when done - not native to the file - its just here to test data easily
from parse import parse

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  t = Node() # single node tree to start
  target_poss = {} # all possible target values
  no_attribs = True # whether or not attributes is empty

  # how many times each Class variable appears
  for i in examples: # going through each dictionary
    key = i["Class"] # checking the "Class" attribute of each example
    target_poss[key] += 1; # adding to frequency
    if len(target_poss[key]) > 1:
      # if there are attributes in a dataset, then
      # we can compute it
      no_attribs = False

  if no_attribs:
    return t

  # Decide best split best off information gain
  starting_entropy = get_entropy_from_data(examples, "Class", is_first_node=True)

  attribute_to_split: tuple = choose_attribute_split(examples, starting_entropy)

  # root_node = Node(first_split[0], None, examples)
  
  max_keys = dict_max(entropy)
  
  if len(max_keys) > 1:
    # if there is more than one max key
    # just use given default
    t.set_label(default)
  else:
    t.set_label(max_keys[0])
    
  return t

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


### EJ Functions ###

def get_attribute_data(data: list, attribute_name: str, target="Class"):
  """ 
  Returns a dictionary of dictionaries with each  from the attribute as keys
  and the values being dictionaries with targets as keys and their counts as values along with # of observations
  E.G. "Color" attribute from mushroom example -> {"red": {"Total": 2, "toxic": 1, "eatable": 1}, "green": ...}

  Parameters:
  data: list of dictionaries (what parse() outputs)
  attribute_name: Name of attribute to extract data of
  target: Final column with result (all data sets use "Class" as the column header for targets)
  """

  attribute_dict = {}
  attribute_data = []

  try:
    # For each row in data, makes tuple of that row's attribute and target 
    # i.e. [("red", "eatable"), ("brown", "toxic"), ...]
    attribute_data = list(zip([row[attribute_name] for row in data],
                              [row[target] for row in data]))
  
  except Exception as e:
    print(f"Error occured: {e}")
    return


  for cur_item, cur_target in attribute_data:
    item_exists = cur_item in attribute_dict

    # Check if item is in dictionary yet
    if not item_exists: 
      # Initializing dict to hold targets and their counts for the attribute
      attribute_dict[cur_item] = {"Total": 1, cur_target: 1}
    else:
      target_dict = attribute_dict[cur_item]

      # Ensure we count values correctly with dictionary check
      if not cur_target in target_dict:
        target_dict[cur_target] = 1
        # Need to keep track of attributes total count for entropy calculation
        target_dict["Total"] += 1

      else:
        target_dict[cur_target] += 1
        target_dict["Total"] += 1
  
  return attribute_dict, len(attribute_data)


# get_entropy(...) -> Calculates entropy of data with first-node handling
#
# parameter: attribute_dict -> dict of dicts with attributes as keys and dicts w/ targets as keys
# parameter: num_observations -> total # of values in data for entropy calc
# parameter: is_first_node -> used to handle (different) instructions for calculation if this is the first node
def get_entropy(attribute_dict: dict, num_observations: int, is_first_node: False):

  attribute_entropy = 0.0


  for item in attribute_dict:
    target_dict = attribute_dict[item]

    # Iterating over all attributes, so must handle current item entropy while at it
    item_entropy = 0.0

    for cur_target in target_dict:
      # 'Total' is included in every target_dict, but not needed until later
      if(cur_target == "Total"): continue

      # First node entropy only uses "Class" attribute (target) data, so it requires different instructions
      if is_first_node:
        target_probability = target_dict["Total"] / num_observations
        attribute_entropy += (target_probability * math.log2(target_probability))

      target_probability = (target_dict[cur_target] / target_dict["Total"])
      item_entropy += target_probability * math.log2(target_probability)

    # Must sum values of targets entropy before multiplying by neg. 1
    item_entropy *= -1
    item_probability = (attribute_dict[item]["Total"]) / num_observations


    attribute_entropy += (item_entropy * item_probability)

  # First node entropy uses different equation than entropy of entire attribute, so multiply neg. 1 to correct
  if is_first_node:
    return item_entropy * -1
  
  return item_entropy


# get_entropy_from_data(...) -> Extrapolates different functions to more easily get entropy of attribute with
#                               broader data set
# This is pretty straightforward, so I won't bother with parameters :-|
def get_entropy_from_data(data: list, attribute_name: str, is_first_node: False):

  attribute_dict = get_attribute_data(data, attribute_name)[0]
  num_observations = get_attribute_data(data, attribute_name)[1]

  entropy = get_entropy(attribute_dict, num_observations, is_first_node)

  return entropy


# get_information_gain(...) -> Calculates information gain for the current node
#
# parameters: Both are super straightforward again so I'm not gonna bother :-\
def get_information_gain(parent_entropy: float, node_entropy: float) -> float:
  return parent_entropy - node_entropy



def get_attributes(data) -> List[str]:  # typing is not like a normal thing in Python,
                                        # so we would need to import Typing package
  attributes = list([observation for observation in data][0].keys())
  attributes.remove("Class")
  return attributes


#TODO: This algorithim will only pick a new attribute if its info gain is GREATER 
#TODO: than the previous attribute ergo it doesn't deal with ties which could be good
#TODO: for getting the "best" tree
def choose_attribute_split(data: list, parent_entropy: float):
  """
  Finds entropy of existing attributes and splits based off highest information gain



  Parameters:
  
  """
  # Setting default values for first element of loop -> tuple(attribute_name, info_gain, entropy)
  best_attribute = ("foo", -1, None)
  for cur_attribute in get_attributes(data):
    cur_entropy = get_entropy_from_data(data, cur_attribute, is_first_node=False)

    cur_info_gain = get_information_gain(parent_entropy, cur_entropy)
    if cur_info_gain > best_attribute[1]:
      best_attribute = (cur_attribute, cur_info_gain, cur_entropy)

  return best_attribute


def split_data(original_data, attribute_to_split):

  possible_attributes = {}

  for observation in original_data:
    # {"color": "red", "points": "yes", "Size": "Small", "Eatiablility": "Eatable"}

    cur_item = observation[attribute_to_split] # -> "red"
    del observation[attribute_to_split]

    if not cur_item in possible_attributes:
      # Remove the attribute we are splitting on
      possible_attributes[cur_item] = [observation]
    else:
      possible_attributes[cur_item].append(observation)

  return possible_attributes
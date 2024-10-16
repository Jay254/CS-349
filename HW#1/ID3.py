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

  # Decide best split best off information gain
  starting_entropy = get_entropy_from_data(examples, "Class", is_first_node=True)

  feature_to_split: tuple = choose_feature_split(examples, starting_entropy)

  # root_node = Node(first_split[0], None, examples)

  



  return first_split


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

def get_feature_data(data: list, feature_name: str, target="Class"):
  """ 
  Returns a dictionary of dictionaries with each attribute from the feature as keys
  and the values being dictionaries with targets as keys and their counts as values along with # of observations
  E.G. "Color" feature from mushroom example -> {"red": {"Total": 2, "toxic": 1, "eatable": 1}, "green": ...}

  Parameters:
  data: list of dictionaries (what parse() outputs)
  feature_name: Name of feature to extract data of
  target: Final column with result (all data sets use "Class" as the column header for targets)
  """

  feature_dict = {}
  feature_data = []

  try:
    # For each row in data, makes tuple of that rows attribute and target 
    # i.e. [("red", "eatable"), ("brown", "toxic"), ...]
    feature_data = list(zip([row[feature_name] for row in data],
                              [row[target] for row in data]))
  
  except Exception as e:
    print(f"Error occured: {e}")
    return


  for cur_atrbt, cur_target in feature_data:
    atrbt_exists = cur_atrbt in feature_dict

    # Check if attribute is in dictionary yet
    if not atrbt_exists: 
      # Initializing dict to hold targets and their counts for the attribute
      feature_dict[cur_atrbt] = {"Total": 1, cur_target: 1}
    else:
      target_dict = feature_dict[cur_atrbt]

      # Ensure we count values correctly with dictionary check
      if not cur_target in target_dict:
        target_dict[cur_target] = 1
        # Need to keep track of attributes total count for entropy calculation
        target_dict["Total"] += 1

      else:
        target_dict[cur_target] += 1
        target_dict["Total"] += 1
  
  return feature_dict, len(feature_data)


# get_entropy(...) -> Calculates entropy of data with first-node handling
#
# parameter: feature_dict -> dict of dicts with attributes as keys and dicts w/ targets as keys
# parameter: num_observations -> total # of values in data for entropy calc
# parameter: is_first_node -> used to handle (different) instructions for calculation if this is the first node
def get_entropy(feature_dict: dict, num_observations: int, is_first_node: False):

  feature_entropy = 0.0


  for attribute in feature_dict:
    target_dict = feature_dict[attribute]

    # Iterating over all attributes, so must handle current attribute entropy while at it
    attr_entropy = 0.0

    for cur_target in target_dict:
      # 'Total' is included in every target_dict, but not needed until later
      if(cur_target == "Total"): continue

      # First node entropy only uses "Class" feature (target) data, so it requires different instructions
      if is_first_node:
        target_probability = target_dict["Total"] / num_observations
        feature_entropy += (target_probability * math.log2(target_probability))

      target_probability = (target_dict[cur_target] / target_dict["Total"])
      attr_entropy += target_probability * math.log2(target_probability)

    # Must sum values of targets entropy before multiplying by neg. 1
    attr_entropy *= -1
    attr_probability = (feature_dict[attribute]["Total"]) / num_observations


    feature_entropy += (attr_entropy * attr_probability)

  # First node entropy uses different equation than entropy of entire feature, so multiply neg. 1 to correct
  if is_first_node:
    return feature_entropy * -1
  
  return feature_entropy


# get_entropy_from_data(...) -> Extrapolates differnet functions to more easily get entropy of feature with
#                               broader data set
# This is pretty straightforward, so I won't bother with parameters :-|
def get_entropy_from_data(data: list, feature_name: str, is_first_node: False):

  feature_dict = get_feature_data(data, feature_name)[0]
  num_observations = get_feature_data(data, feature_name)[1]

  entropy = get_entropy(feature_dict, num_observations, is_first_node)

  return entropy


# get_informaiton_gain(...) -> Calculates information gain for the current node
#
# parameters: Both are super straightforward again so I'm not gonna bother :-\
def get_information_gain(parent_entropy: float, node_entropy: float) -> float:
  return parent_entropy - node_entropy



def get_features(data) -> [str]:
  features = list([observation for observation in data][0].keys())
  features.remove("Class")
  return features


#TODO: This algorithim will only pick a new feature if its info gain is GREATER 
#TODO: than the previous feature ergo it doesn't deal with ties which could be good
#TODO: for getting the "best" tree
def choose_feature_split(data: list, parent_entropy: float):
  """
  Finds entropy of existing features and splits based off highest information gain



  Parameters:
  
  """
  # Setting default values for first element of loop -> tuple(feature_name, info_gain, entropy)
  best_feature = ("foo", -1, None)
  for cur_feature in get_features(data):
    cur_entropy = get_entropy_from_data(data, cur_feature, is_first_node=False)

    cur_info_gain = get_information_gain(parent_entropy, cur_entropy)
    if cur_info_gain > best_feature[1]:
      best_feature = (cur_feature, cur_info_gain, cur_entropy)

  return best_feature


def split_data(original_data, feature_to_split):

  possible_attributes = {}

  for observation in original_data:
    # {"color": "red", "points": "yes", "Size": "Small", "Eatiablility": "Eatable"}

    cur_attr = observation[feature_to_split] # -> "red"
    del observation[feature_to_split]

    if not cur_attr in possible_attributes:
      # Remove the feature we are splitting on
      possible_attributes[cur_attr] = [observation]
    else:
      possible_attributes[cur_attr].append(observation)

  return possible_attributes




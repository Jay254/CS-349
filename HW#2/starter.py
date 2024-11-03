# returns Euclidean distance between vectors and b
def euclidean(a, b):
    # read_data() doesn't convert values from csv to floats -> must do so for calculations
    a_converted = [float(x) for x in a]
    b_converted = [float(x) for x in b]

    total = 0
    try:
        total = sum([(x - y) ** 2 for x, y in zip(a_converted, b_converted)])

    except Exception as e:
        # Different input lengths
        print(e)

    # Not sure if allowed to import math -> power of 1/2 is the same thing
    return float(total ** (1 / 2))

# returns Cosine Similarity between vectors and b
def cosim(a, b):
    # read_data() doesn't convert values from csv to floats -> must do so for calculations
    a_converted = [float(x) for x in a]
    b_converted = [float(x) for x in b]
    
    numerator =  0
    a_magnitude = 0
    b_magnitude = 0
    try:
        # calculate dot product
        numerator = sum([x * y for x, y in zip(a_converted, b_converted)])
        a_magnitude = sum([x ** 2 for x in a_converted]) ** (1 / 2)
        b_magnitude = sum([y ** 2 for y in b_converted]) ** (1 / 2)

    except Exception as e:
        # Different input lengths
        print(e)

    # avoid division by 0 -> return 0
    if a_magnitude == 0 or b_magnitude == 0:
        return 0

    return float(numerator / (a_magnitude * b_magnitude))

# returns Hamming distance between vectors and b
def hamming(a, b):
    # read_data() doesn't convert values from csv to floats -> must do so for calculations
    a_converted = [float(x) for x in a]
    b_converted = [float(x) for x in b]

    dist = -1
    try:
        # x_i == y_i -> add 0 to dist, else add 1
        dist = sum([0 if x == y else 1 for x, y in zip(a_converted, b_converted)])
    except Exception as e:
        # Different input lengths
        print(e)

    return float(dist)

# returns Pearson Correkation between vectors and b
def pearson(a, b):
    # This one is more complicated, so i am going to break it down more for easier comprehension
    # read_data() doesn't convert values from csv to floats -> must do so for calculations
    a_converted = [float(x) for x in a]
    b_converted = [float(x) for x in b]
    
    # Pearson distance = 1 - r

    # find r
    # get average of a and b
    a_avg = sum(a_converted) / len(a_converted)
    b_avg = sum(b_converted) / len(b_converted)

    # multiply the i-th element of a-a_avg with b-b_avg
    numerator = sum([(cur_a - a_avg) * (cur_b - b_avg) for cur_a, cur_b in zip(a_converted, b_converted)])  # spaced out for readablility

    # get components of denominator
    a_denom = sum([(cur_a - a_avg) ** 2 for cur_a in a_converted]) ** (1 / 2)
    b_denom = sum([(cur_b - b_avg) ** 2 for cur_b in b_converted]) ** (1 / 2)

    # Put it all together to get r (with 0 division handling)
    if a_denom == 0 or b_denom == 0:
        return None
    r = numerator / (a_denom * b_denom)

    # Final result
    return float(1 - r)

''' KNN functions start '''
# returns a list of labels for the query dataset based upon labeled observations in the dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.

def knn(data,query,metric,k=11):

    # Hyperparameters:
    # k = 11 # -> num neighbors to compare

    # IMPORTANT: read_data() structures mnsit data like so:
    # [label, [observation_data]]
    data_start_idx = 1
    dist_index = 1
    label_index = 0

    # Turn strings into floats to allow arithmetic
    float_data = []
    for observation in data:
        float_data.append([float(observation[label_index]), [float(pixel) for pixel in observation[data_start_idx]]])

    float_query = []
    for observation in query:
        float_query.append([float(pixel) for pixel in observation])


    # Transform data to make program run faster
    transformed_data = [[observation[label_index], knn_data_transformation(observation[data_start_idx])] for observation in float_data]
    transformed_query = [knn_data_transformation(observation) for observation in float_query]
    
    # Keep track of k nearest neigbors for each query
    # -> {query: [(distance1, label), (distance2, label), ...]}

    # Key values must be immutable -> to use queries as keys, put them in tuples
    query_tuple_copy = [tuple(this_query) for this_query in transformed_query]
    closest_neighbors = {cur_query: [] for cur_query in query_tuple_copy}
    
    if metric == "euclidean":
        for observation in transformed_data:
            # print(f"Finished observation: {index}")
            for cur_query in query_tuple_copy: 
                # Label is present in observation data (first element), exclude it in calculation for correct distance
                distance = euclidean(cur_query, observation[data_start_idx])
                # Can insert distance immediatly (no checking) if we don't already have k neigbors
                # NOTE: Converting curquery to tuple so we can use it as a dictionary key
                if len(closest_neighbors[cur_query]) < k:
                    closest_neighbors[cur_query].append((observation[label_index], distance))
                    continue

                # Otherwise find maximum distance element and see if cur distance is closer
                furthest_nbr = max(closest_neighbors[cur_query], key=lambda pair: pair[dist_index])[dist_index] # gets tuple with greatest dist (furthest) and takes the distance

                if distance < furthest_nbr:
                    # remove old value
                    closest_neighbors[cur_query].remove(max(closest_neighbors[cur_query], key=lambda pair: pair[dist_index]))
                    # insert new one
                    closest_neighbors[cur_query].append((observation[label_index], distance))

    elif metric == "cosim":
        for observation in transformed_data:
            # print(f"Finished observation: {index}")
            for cur_query in query_tuple_copy: 
                # Label is present in observation data (first element), exclude it in calculation for correct distance
                distance = cosim(cur_query, observation[data_start_idx])
                # Can insert distance immediatly (no checking) if we don't already have k neigbors
                # NOTE: Converting curquery to tuple so we can use it as a dictionary key
                if len(closest_neighbors[cur_query]) < k:
                    closest_neighbors[cur_query].append((observation[label_index], distance))
                    continue

                # Otherwise find value with lowest similarity
                furthest_nbr = min(closest_neighbors[cur_query], key=lambda pair: pair[dist_index])[dist_index] # gets tuple with LOWEST similarity

                if distance > furthest_nbr:
                    # remove old value
                    closest_neighbors[cur_query].remove(min(closest_neighbors[cur_query], key=lambda pair: pair[dist_index]))
                    # insert new one
                    closest_neighbors[cur_query].append((observation[label_index], distance))
    else:
        print(f"ERROR: Invalid distance metric: {metric}")
        return
    
    # Done with getting nearest neighbors -> find mode label for each query
    labels = knn_get_mode_label(closest_neighbors)

    # print(f"Time to run (seconds): {time.time() - start_time}")
    return labels

# Transforms data to simplify and extract most important qualities in order to speed up KNN
# This function assumes 'data' parameter does NOT include the label and only takes in a SINGLE observation
def knn_data_transformation(observation):
    # Hyper parameter
    new_grid_dimension = 7 # IMPORTANT: old_grid_dimension / new_grid_dimension -> must be evenly divisible for equal number pixels per section
    old_grid_dimension = 28
    
    section_dimension = int(old_grid_dimension / new_grid_dimension) # if new_grid_dimension is 7 -> 4x4 pixels in each section
    
    section_averages = {section_num: 0 for section_num in range(0, new_grid_dimension ** 2)}
    
    row_iteration = 0
    segment_start = 0
    row_tracker = section_dimension - 1
    while segment_start < len(observation):
        cur_section = int((segment_start / section_dimension) - row_iteration * new_grid_dimension)
        section_averages[cur_section] += sum(observation[segment_start : segment_start + section_dimension])# IMPORTANT: list splicing is non-inclusive for end-point -> go 1 element further
        segment_start += section_dimension
        
        if segment_start % old_grid_dimension == 0: # signifies onto next row of pixels in og image
                if row_iteration < row_tracker:
                    row_iteration += 1
                else:
                    row_tracker += section_dimension - 1
    
    # Done with sectioning pixels together -> find average value (black/white) for each section
    transformed_data = [section_sum / (section_dimension ** 2) for section_sum in section_averages.values()]
    return transformed_data

# nbrs_dict -> dict on k length
def knn_get_mode_label(nbrs_dict: dict):
    final_labels = []
    for query in nbrs_dict:
        cur_freq_dict = {}
        for label, _ in nbrs_dict[tuple(query)]:
            if label in cur_freq_dict:
                cur_freq_dict[label] += 1
            else:
                cur_freq_dict[label] = 1
        # Get most frequent label
        # IMPORTANT: when there are ties between most common labels,
        #            max() simply chooses whichever element came first
        result_label = max(cur_freq_dict, key=cur_freq_dict.get)
        final_labels.append(result_label)
        
    return final_labels

# Displays model accuracy on a testing set and what values it got wrong most often 
# predictions -> list of labels from model's predictions
# actual_values -> objective values to be compared against
def display_incorrect_results(predictions: list, actual_values: list):


    count = 1
    correct = 0
    incorrect = 0
    confusion_matrix = {str(number): {str(num_2): 0 for num_2 in range(0,10)} for number in range(0, 10)}
    for prediction, actual in zip(predictions, [observation[0] for observation in actual_values]):
        if prediction == actual:
            # print(f"{count}.) Prediction: {prediction} | Actual: {actual} -> Correct")
            correct += 1
        else:
            # print(f"{count}.) Prediction: {prediction} | Actual: {actual} -> Incorrect")
            incorrect += 1
        
        confusion_matrix[actual][prediction] += 1

        count += 1
    print(f"\nAccuracy: {(correct / (incorrect + correct) * 100)}%\n")
    print("Of the incorrect guesses, the percent that were from each number is...")
    # for number in incorrect_tracker:
    #     print(f"\t{number} was {round((incorrect_tracker[number] / incorrect) * 100, 2)}% of incorrect guesses")

    for actual in confusion_matrix:
        num_wrong_guesses = sum(list(confusion_matrix[actual].values())) - confusion_matrix[actual][actual]
        print(f"Actual label: {actual} | Incorrectly labeled {num_wrong_guesses} times")
        for wrong_guess in confusion_matrix[actual]:
            print(f"\tGuessed {wrong_guess} -> {confusion_matrix[actual][wrong_guess]} times")


    return (correct / (incorrect + correct) * 100)

''' KNN functions end '''


def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    show('valid.csv','pixels')
    
if __name__ == "__main__":
    main()
    

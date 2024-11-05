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

def knn(data,query,metric):
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
     

    # Hyperparameters:
    k = 11 # -> num neighbors to compare
    
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

# returns a list of labels for the query dataset based upon observations in the dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(data,query,metric):
    # Convert input data to float format, ignoring labels
    float_data = [list(map(float, observation[1])) for observation in data]

    # Hyper-parameter
    k = 3  # Number of clusters

    max_iterations = 100 # Maximum number of iterations to prevent infinite loops

    # Step 1: Initialize centroids randomly from the data points
    centroids = random.sample(float_data, k)

    # Loop for a maximum number of iterations to refine clusters
    for iteration in range(max_iterations):
        # Step 2: Assign each observation to the closest centroid
        clusters = {i: [] for i in range(k)}
        for observation in float_data:
            # Determine the closest centroid based on the specified metric
            if metric == "euclidean":
                distances = [euclidean(observation, centroids[i]) for i in range(k)]
                closest_centroid_index = distances.index(min(distances))
            elif metric == "cosim":
                similarities = [cosim(observation, centroids[i]) for i in range(k)]
                closest_centroid_index = similarities.index(max(similarities))
            else:
                raise ValueError(f"ERROR: Invalid distance metric: {metric}")
            
            # Assign the observation to the corresponding cluster
            clusters[closest_centroid_index].append(observation)

        # Step 3: Update centroids based on the current clusters
        new_centroids = []
        for i in range(k):
            if clusters[i]:  # Check if cluster is not empty
                # Calculate the new centroid as the mean of all points in the cluster
                dimension_count = len(clusters[i][0])  # Get the number of dimensions
                sums = [0] * dimension_count  # Initialize sums for each dimension
                for point in clusters[i]:
                    for dim in range(dimension_count):
                        sums[dim] += point[dim]  # Accumulate the sums
                new_centroid = [s / len(clusters[i]) for s in sums]  # Average each dimension
            else:
                # If a cluster has no points, keep the old centroid
                new_centroid = centroids[i]  # Keep the old centroid if no points are assigned
            new_centroids.append(new_centroid)

        # Check for convergence: if centroids have not changed, exit loop
        if new_centroids == centroids:
            break
        centroids = new_centroids # Update for next iteration

    # Step 4: Predict labels for the query dataset based on final centroids
    labels = []
    for query_point in query:
        # Determine the closest centroid for each query point
        if metric == "euclidean":
            distances = [euclidean(query_point, centroids[i]) for i in range(k)]
            closest_centroid_index = distances.index(min(distances))
        elif metric == "cosim":
            similarities = [cosim(query_point, centroids[i]) for i in range(k)]
            closest_centroid_index = similarities.index(max(similarities))
        # Add the predicted label for the query point
        labels.append(closest_centroid_index)
    return(labels) # Return the list of predicted labels for the query points

def get_user_ratings(filename):
    """Get ratings dictionary from a file"""
    ratings = {}
    with open(filename, "r") as f:
        next(f)  # skip header
        for line in f:
            tokens = line.split()
            movie_id = tokens[1]
            rating = float(tokens[2])
            ratings[movie_id] = rating
    return ratings


def collaborative_filter(movielens_file: str, user_file: str, k: int, m: int):
    # k is the number of similar users to consider
    # m is the number of movies to recommend
    if (k <= 0) or (m <= 0):
        raise ValueError("Enter valid k and n values.")

    # Get target users' ratings from selected file
    target_ratings = get_user_ratings(user_file)

    # Get other users' ratings from movielens file
    other_users = {}
    with open(movielens_file, "r") as f:
        next(f)  # skip header
        for line in f:
            tokens = line.split()
            userID = tokens[0]
            movie_id = tokens[1]
            rating = float(tokens[2])
            if userID not in other_users:
                other_users[userID] = {}
            other_users[userID][movie_id] = rating

    # Find similar users
    similar_users = {}
    for userID in other_users:
        similar_movies = []
        # Find movies that both users have rated
        for movie_id in target_ratings:
            if movie_id in other_users[userID]:
                similar_movies.append(movie_id)

        if similar_movies:  # Only consider users with common movies
            vector_a = []
            vector_b = []
            for movie_id in similar_movies:
                vector_a.append(target_ratings[movie_id])
                vector_b.append(other_users[userID][movie_id])
            # Calculate similarity score
            similarity = cosim(vector_a, vector_b)
            similar_users[userID] = similarity

    # Get top k similar users
    sorted_similar_users = sorted(
        similar_users.items(), key=lambda x: x[1], reverse=True
    )
    k_similar_users = sorted_similar_users[:k]

    # Get recommendations from similar users
    recommendations = {}
    for userID, similarity in k_similar_users:
        for movie_id, rating in other_users[userID].items():
            if movie_id not in target_ratings:  # Only recommend unwatched movies
                if movie_id not in recommendations:
                    recommendations[movie_id] = []
                recommendations[movie_id].append(rating * similarity)

    # Calculate weighted average scores
    for movie_id in recommendations:
        recommendations[movie_id] = sum(recommendations[movie_id]) / len(
            recommendations[movie_id]
        )

    # Sort by score and get top n
    sorted_recommendations = sorted(
        recommendations.items(), key=lambda x: x[1], reverse=True
    )

    recommended_movies = []
    for movie_id, _ in sorted_recommendations[:m]:
        recommended_movies.append(movie_id)

    return recommended_movies


def calculate_tests(recommendations: list, actual_ratings: dict):
    # note: True positive is a movie that was recommended and appears in actual ratings.
    # False positive is a movie that was recommended but does not appear in actual ratings.
    # False negative is a movie that appears in actual ratings but was not recommended.

    if not recommendations or not actual_ratings:
        return 0.0, 0.0, 0.0

    # Get sets for easier comparison
    recommended_set = set(recommendations)
    actual_set = set(actual_ratings.keys())

    # Calculate metrics
    true_positives = len(recommended_set & actual_set)  # intersection
    false_positives = len(
        recommended_set - actual_set
    )  # in recommendations but not in actual ratings
    false_negatives = len(
        actual_set - recommended_set
    )  # in actual ratings but not in recommendations

    # Calculate precision and recall
    if (true_positives + false_positives) == 0.0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    if (true_positives + false_negatives) == 0.0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    # Calculate F1 score
    if (precision + recall) == 0.0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

def collaborative_filter_plus(movielens_file: str, user_file: str, k: int, m: int):
    """
    # Finds similar users to a specified user ID and recommends movies based on ratings, demographics, and genre preferences.
    # movielens_file -> main dataset file
    # user_file -> target user ratings file
    # k -> number of similar users we want to consider
    # m -> number of movies to recommend to user
    """
    if (k <= 0) or (m <= 0):
        raise ValueError("Enter valid k and m values.")

    #analyze user rating patterns
    def calculate_rating_patterns(ratings):
        if not ratings:
            return 0, 0, {}
        
        values = list(ratings.values())
        avg = sum(values) / len(values)
        var = sum((x - avg) ** 2 for x in values) / len(values)
        dist = {i: sum(1 for x in values if i-0.5 <= x < i+0.5) / len(values) 
               for i in range(1, 6)}
        return avg, var, dist

    #get similarity between two user rating patterns
    def rating_pattern_similarity(pattern1, pattern2):
        avg1, var1, dist1 = pattern1
        avg2, var2, dist2 = pattern2
        
        # rating distributions
        dist_sim = sum(min(dist1.get(i, 0), dist2.get(i, 0)) for i in range(1, 6))
        
        # Compare rating behaviors
        behavior_sim = 1 - (abs(avg1 - avg2) / 4)  # Normalize by max possible difference
        variance_sim = 1 - (abs(var1 - var2) / 16)  # Max variance possible == 16
        
        return 0.4 * dist_sim + 0.3 * behavior_sim + 0.3 * variance_sim

    #get gen prefs based on ratings
    def calculate_genre_preferences(ratings_with_genres):
        genre_scores = {}
        genre_counts = {}
        
        for movie_id, (rating, genre) in ratings_with_genres.items():
            if genre not in genre_scores:
                genre_scores[genre] = 0
                genre_counts[genre] = 0
            genre_scores[genre] += rating
            genre_counts[genre] += 1
        
        #average rating per genre
        preferences = {genre: genre_scores[genre] / genre_counts[genre] 
                      for genre in genre_scores}
        
        # normalize those preferences
        total = sum(preferences.values())
        if total > 0:
            preferences = {g: s/total for g, s in preferences.items()}
            
        return preferences

    # load target user data
    target_ratings = {}
    target_info = None
    target_movies = {}  # Store both rating and genre
    
    with open(user_file, "r") as f:
        next(f) 
        for line in f:
            tokens = line.strip().split('\t')
            movie_id = tokens[1]
            rating = float(tokens[2])
            genre = tokens[4]
            
            target_ratings[movie_id] = rating
            target_movies[movie_id] = (rating, genre)
            
            #store demo info (only once)
            if target_info is None:
                target_info = {
                    'age': int(tokens[5]),
                    'gender': tokens[6],
                    'occupation': tokens[7]
                }

    #target user rating patterns/prefs
    target_patterns = calculate_rating_patterns(target_ratings)
    target_preferences = calculate_genre_preferences(target_movies)

    # other users' data
    users_data = {}
    with open(movielens_file, "r") as f:
        next(f)  # skip header
        for line in f:
            tokens = line.strip().split('\t')
            userID = tokens[0]
            movie_id = tokens[1]
            rating = float(tokens[2])
            genre = tokens[4]
            
            if userID not in users_data:
                users_data[userID] = {
                    'ratings': {},
                    'movies': {},
                    'demographics': {
                        'age': int(tokens[5]),
                        'gender': tokens[6],
                        'occupation': tokens[7]
                    }
                }
            
            users_data[userID]['ratings'][movie_id] = rating
            users_data[userID]['movies'][movie_id] = (rating, genre)

    # get similar users
    similar_users = {}
    min_common_movies = 5  # min threshold for common movies
    
    for userID, user_data in users_data.items():
        try:
            # calculate rating similarity
            common_movies = set(target_ratings.keys()) & set(user_data['ratings'].keys())
            if len(common_movies) < min_common_movies:
                continue
                
            target_vector = [target_ratings[m] for m in common_movies]
            user_vector = [user_data['ratings'][m] for m in common_movies]
            rating_sim = cosim(target_vector, user_vector)
            
            if rating_sim <= 0.1:  # min similarity threshold
                continue

            # rating pattern similarity
            user_patterns = calculate_rating_patterns(user_data['ratings'])
            pattern_sim = rating_pattern_similarity(target_patterns, user_patterns)

            # genre pref similarity
            user_preferences = calculate_genre_preferences(user_data['movies'])
            
            # comparing genre similarity using common genres
            common_genres = set(target_preferences.keys()) & set(user_preferences.keys())
            if common_genres:
                genre_vector1 = [target_preferences.get(g, 0) for g in common_genres]
                genre_vector2 = [user_preferences.get(g, 0) for g in common_genres]
                genre_sim = cosim(genre_vector1, genre_vector2)
            else:
                genre_sim = 0

            # demo similarity
            age_diff = abs(target_info['age'] - user_data['demographics']['age'])
            age_sim = max(0, 1 - age_diff/50)  # Linear decay up to 50 years difference
            gender_sim = 1 if target_info['gender'] == user_data['demographics']['gender'] else 0
            occ_sim = 1 if target_info['occupation'] == user_data['demographics']['occupation'] else 0
            demo_sim = 0.5 * age_sim + 0.3 * gender_sim + 0.2 * occ_sim

            # weighting based on common movies and rating similarity
            common_weight = min(1.0, len(common_movies) / 50)
            rating_base_weight = 0.6
            rating_weight = rating_base_weight + (0.2 * common_weight)
            remaining_weight = 1.0 - rating_weight
            
            # Combined simillarity metrics
            combined_sim = (
                rating_sim * rating_weight +
                pattern_sim * (remaining_weight * 0.4) +
                genre_sim * (remaining_weight * 0.4) +
                demo_sim * (remaining_weight * 0.2)
            )
            
            similar_users[userID] = combined_sim

        except Exception as e:
            continue

    # get recommendations from top k similar users
    recommendations = {}
    for userID, similarity in sorted(similar_users.items(), key=lambda x: x[1], reverse=True)[:k]:
        user_ratings = users_data[userID]['ratings']
        for movie_id, rating in user_ratings.items():
            if movie_id not in target_ratings:
                if movie_id not in recommendations:
                    recommendations[movie_id] = []
                # Include both rating and similarity in the weighted calculation
                recommendations[movie_id].append((rating, similarity))

    # final scores emphasising on higher ratings
    final_scores = {}
    for movie_id, ratings_sims in recommendations.items():
        if ratings_sims:
            #square similarities to give more weight to more similar users
            weighted_sum = sum(rating * (similarity ** 2) for rating, similarity in ratings_sims)
            weight_sum = sum(similarity ** 2 for _, similarity in ratings_sims)
            final_scores[movie_id] = weighted_sum / weight_sum if weight_sum > 0 else 0

    #top m recommendations
    sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, _ in sorted_recommendations[:m]]


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
    

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

def collaborative_filter(file_name: str, selectedUserID: int, k: int, n: int):

    # Finds similar users to specified userID, recommends movies based on k-similar users
    # File_name -> name of file
    # userID -> ID of user that we want to recommend movies to
    # k -> number of similar users we want to consider
    # n -> number of movies to recommend to user

    # build dictionary with userIDs and their rating for each movie
    user_ratings = {}
    with open(file_name, "r") as f:
        for line in f:
            line = line.split()
            userID = line[0]
            movie_id = line[1]
            rating = line[2]
            if userID not in user_ratings:
                user_ratings[userID] = {}
            user_ratings[userID][movie_id] = rating

    target_ratings = user_ratings[selectedUserID]

    similar_users = {}
    for userID in user_ratings:
        if userID == selectedUserID:
            continue
        similar_movies = []
        # Find similar movies between selected user and other users
        for movie_id in target_ratings:
            if movie_id in user_ratings[userID]:
                similar_movies.append(movie_id)

        vector_a = []
        vector_b = []
        for movie_id in similar_movies:
            vector_a.append(target_ratings[movie_id])
            vector_b.append(user_ratings[userID][movie_id])
        # Use cosim to find similarity "score"
        similarity = cosim(vector_a, vector_b)
        similar_users[userID] = similarity

    # Sort in descending order
    sorted_similar_users = similar_users.sort(key=lambda item: item[1], reverse=True)
    k_similar_users = sorted_similar_users[:k]

    recommendations = {}
    for userID, similarity in k_similar_users:
        for movie_id, rating in user_ratings[userID].items():
            if movie_id not in target_ratings:
                recommendations[movie_id] = rating * similarity

    # Need to standardize ratings
    for movie_id, rating in recommendations.items():
        recommendations[movie_id] = sum(rating) / len(rating)

    # Sort recomendations according to score
    sorted_recommendations = recommendations.sort(key=lambda item: item[1], reverse=True)
    list_of_recommendations = []
    count = 0
    for movie_id, rating in sorted_recommendations:
        # Ensure recommendation count doesn't exceed amount specified in function
        if count >= n:
            break
        # Append movie ids
        list_of_recommendations.append(movie_id)
        count += 1

    return list_of_recommendations

def collaborative_filter_plus(file_name: str, selectedUserID: int, k: int, n: int):

    # Finds similar users to a specified user ID and recommends movies based on ratings, demographics, and genre preferences.
    # File_name -> name of file
    # userID -> ID of user that we want to recommend movies to
    # k -> number of similar users we want to consider
    # n -> number of movies to recommend to user
    
    # demographic similarity
    def demosim(user1_demo, user2_demo):
        age_sim = 1 - abs(int(user1_demo['age']) - int(user2_demo['age']))/100 #normalize age similarity
        gender_sim = 1 if user1_demo['gender'] == user2_demo['gender'] else 0
        occupation_sim = 1 if user1_demo['occupation'] == user2_demo['occupation'] else 0
        
        # chhosen weights for demographic features
        weights = {'age': 0.4, 'gender': 0.3, 'occupation': 0.3}
        return (age_sim * weights['age'] + 
                gender_sim * weights['gender'] + 
                occupation_sim * weights['occupation'])

    #genre preference similarity
    def gensim(user_movies):
        genre_ratings = {}
        genre_counts = {}
        
        for movie_id, (rating, genre) in user_movies.items():
            if genre not in genre_ratings:
                genre_ratings[genre] = []
            genre_ratings[genre].append(float(rating))
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
        # avg rating per genre
        genre_pref = {genre: sum(ratings)/len(ratings) 
                           for genre, ratings in genre_ratings.items()}
        return genre_pref, genre_counts

    # parse user data
    users_data = {}  # all user info
    with open(file_name, 'r') as f:
        for line in f:
            line = line.split()
            userID = line[0]
            movie_id = line[1]
            rating = line[2]
            #title = line[3] not needed rn
            genre = line[4]
            age = line[5]
            gender = line[6]
            occupation = line[7]    
            if userID not in users_data:
                users_data[userID] = {
                    'demographics': {'age': age, 'gender': gender, 'occupation': occupation},
                    'movies': {}  # will store movie_id: (rating, genre) <- as a pair, #388 below
                }
            
            # add movie's rating and genre
            users_data[user_id]['movies'][movie_id] = (rating, genre)

    # genre preferences for all users
    for user_id in users_data:
        genre_prefs, genre_counts = gensim(users_data[user_id]['movies'])
        users_data[user_id]['genre_prefs'] = genre_prefs
        users_data[user_id]['genre_counts'] = genre_counts

    # get similar users
    similar_users = {}
    target_user = users_data[selectedUserID]
    
    for user_id, user_data in users_data.items():
        if user_id == selectedUserID:#skip the user we getting recs for
            continue
            
        # calculate rating similarity
        similar_movies = set(target_user['movies'].keys()) & set(user_data['movies'].keys())
        if similar_movies:
            target_ratings = [float(target_user['movies'][m][0]) for m in similar_movies]
            user_ratings = [float(user_data['movies'][m][0]) for m in similar_movies]
            rating_sim = cosim(target_ratings, user_ratings)
        else:
            rating_sim = 0
            
        # get demographic similarity
        demo_sim = demosim(
            target_user['demographic'],
            user_data['demographic']
        )
            
        # get genre pref similarity
        genre_vectors = []
        for user in [target_user, user_data]:
            genre_vec = []
            all_genres = set(user['genre_counts'].keys())
            for genre in all_genres:
                genre_vec.append(user['genre_counts'].get(genre, 0))
            genre_vectors.append(genre_vec)
        
        genre_sim = cosim(genre_vectors[0], genre_vectors[1])
            
        # Combine those similarities with specific weights (based on ranked intuitive importance)
        weights = {'rating': 0.5, 'demographic': 0.3, 'genre': 0.2} #add to 1 for normalization
        combined_sim = (rating_sim * weights['rating'] + 
                       demo_sim * weights['demographic'] + 
                       genre_sim * weights['genre'])
    
        similar_users[user_id] = combined_sim

    # now get recommendations from top k similar users
    sorted_similar_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)
    k_similar_users = sorted_similar_users[:k]
    
    # find movies rated by similar users but not by target user
    recommendations = {}
    target_movies = set(target_user['movies'].keys())
    
    for userID, similarity in k_similar_users:
        user_movies = users_data[userID]['movies']
        for movie_id, (rating, _) in user_movies.items():
            if movie_id not in target_movies:
                if movie_id not in recommendations:
                    recommendations[movie_id] = []
                recommendations[movie_id].append(float(rating) * similarity)

    # standardize scores
    for movie_id, scores in recommendations.items():
        recommendations[movie_id] = sum(scores) / len(scores)

    # Sort recommendations according to score
    sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)

    # top n movie recommendations
    list_of_recommendations = []
    count = 0
    for movie_id, rating in sorted_recommendations:
        # Ensure recommendation count doesn't exceed amount specified in function
        if count >= n:
            break
        # Append movie ids
        list_of_recommendations.append(movie_id)
        count += 1

    return list_of_recommendations


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
    

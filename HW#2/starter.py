# returns Euclidean distance between vectors and b
def euclidean(a, b):
    try:
        total = sum([(x - y) ** 2 for x, y in zip(a, b)])

    except Exception as e:
        # Different input lengths
        print(e)

    # Not sure if allowed to import math -> power of 1/2 is the same thing
    return float(total ** (1 / 2))


# returns Cosine Similarity between vectors and b
def cosim(a, b):
    try:
        # calculate dot product
        numerator = sum([x * y for x, y in zip(a, b)])
        a_magnitude = sum([x ** 2 for x in a]) ** (1 / 2)
        b_magnitude = sum([y ** 2 for y in b]) ** (1 / 2)

    except Exception as e:
        # Different input lengths
        print(e)

    # avoid division by 0 -> return 0
    if a_magnitude == 0 or b_magnitude == 0:
        return 0

    return float(numerator / (a_magnitude * b_magnitude))


# returns Hamming distance between vectors and b
def hamming(a, b):
    try:
        # x_i == y_i -> add 0 to dist, else add 1
        dist = sum([0 if x == y else 1 for x, y in zip(a, b)])
    except Exception as e:
        # Different input lengths
        print(e)

    return float(dist)


# returns Pearson Correkation between vectors and b
def pearson(a, b):
    # This one is more complicated, so i am going to break it down more for easier comprehension
    # Pearson distance = 1 - r

    # find r
    # get average of a and b
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)

    # multiply the i-th element of a-a_avg with b-b_avg
    numerator = sum([(cur_a - a_avg) * (cur_b - b_avg) for cur_a, cur_b in zip(a, b)])  # spaced out for readablility

    # get components of denominator
    a_denom = sum([(cur_a - a_avg) ** 2 for cur_a in a]) ** (1 / 2)
    b_denom = sum([(cur_b - b_avg) ** 2 for cur_b in b]) ** (1 / 2)

    # Put it all together to get r (with 0 division handling)
    if a_denom == 0 or b_denom == 0:
        return None
    r = numerator / (a_denom * b_denom)

    # Final result
    return float(1 - r)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    return(labels)

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
    

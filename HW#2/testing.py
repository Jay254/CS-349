from starter import read_data, kmeans
# import numpy

def testKMeans():
    train = read_data('mnist_train.csv')
    valid_raw = read_data('mnist_valid.csv')
    valid = []
    for obs in valid_raw:
        valid.append(obs[1])
    test_raw = read_data('mnist_test.csv')
    test = []
    for obs in test_raw:
        test.append(obs[1])
    
    values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 20, 40, 50]
    for i in values:
        labels = kmeans(train, test, 'cosim', k=i)
        test_labels = [i[0] for i in test_raw]
        print("accuracy @ ", i, ": ", accuracy(labels, test_labels))
    # print(labels)

    # test_labels = [i[0] for i in test_raw]
    # print(test_labels)
    # print(accuracy(labels, test_labels))


def accuracy(results, labels):
    sum = 0
    for i in range(len(labels)):
        # print("label value: ", labels[i])
        # print("result value: ", int(results[i]))
        if int(labels[i]) - int(results[i]) == 0:
            sum += 1
    # print(sum)
    return (sum / len(results))

def main(): 
    testKMeans()

if __name__ == "__main__":
    main()





















    
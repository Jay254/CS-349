import ID3
from parse import parse
import random
from unit_tests import testPruningOnData
# source: https://machinelearningmastery.com/implement-random-forest-scratch-python/

file = parse("HW#1/candy.data") # parsing file as a randomized set
random.shuffle(file)
print(len(file))
train = file[: len(file) // 2]
print(len(train))
valid = file[len(file) // 2 : 3 * len(file) // 4]
test = file[3 * len(file) // 4 :]
train_set = []

k = 10

sub_len = len(train) // 2 # choosing subsample size to be half of set
print(sub_len)
for i in range(k): # making sure subsets are of the correct length
    sub_set = []
    while len(sub_set) < sub_len:
        index = random.randrange(0, len(train))
        sub_set.append(train[index])
    train_set.append(sub_set)

# rf_train = []
# rf_valid = []
rf_test = []

tree = ID3.ID3([], '1')
print('empty list')

tree = ID3.ID3(train_set[0], '1')
for i in range(len(train_set)):
    # print(len(train_set))
    # print(len(train_set[i]))
    # print("index: ", i)
    # tree = ID3.ID3(train_set[i], '1')
    # print("train_set training")
    # acc = ID3.test(tree, train_set[i])
    # # rf_train.append(acc)
    # # print("train_set appended")

    # acc = ID3.test(tree, valid)
    # # rf_valid.append(acc)
    # print("validation")

    # acc = ID3.test(tree, test)
    # rf_test.append(acc)
    # print("testing")
    testPruningOnData('1', trainFile=None, training=train_set[i], validation=valid, testing=test)

print("tree test: \n")
testPruningOnData('1', trainFile=None, training = train, validation = valid, testing = test)
print("\nrandom forest test:\n")
print(sum(rf_test)/len(rf_test))
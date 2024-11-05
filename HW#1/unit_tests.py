import ID3, parse, random


def testID3AndEvaluate():
	data = [dict(a=1, b=0, Class=1), dict(a=1, b=1, Class=1)]
	tree = ID3.ID3(data, 0)
	if tree != None:
		ans = ID3.evaluate(tree, dict(a=1, b=0))
		if ans != 1:
			print("ID3 test failed.")
		else:
			print("ID3 test succeeded.")
	else:
		print("ID3 test failed -- no tree returned")


def testPruning():
	# data = [dict(a=1, b=1, c=1, Class=0), dict(a=1, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1), dict(a=0, b=0, c=0, Class=1), dict(a=0, b=0, c=1, Class=0)]
	# validationData = [dict(a=0, b=0, c=1, Class=1)]
	data = [
		dict(a=0, b=1, c=1, d=0, Class=1),
		dict(a=0, b=0, c=1, d=0, Class=0),
		dict(a=0, b=1, c=0, d=0, Class=1),
		dict(a=1, b=0, c=1, d=0, Class=0),
		dict(a=1, b=1, c=0, d=0, Class=0),
		dict(a=1, b=1, c=0, d=1, Class=0),
		dict(a=1, b=1, c=1, d=0, Class=0),
	]
	validationData = [
		dict(a=0, b=0, c=1, d=0, Class=1),
		dict(a=1, b=1, c=1, d=1, Class=0),
	]
	tree = ID3.ID3(data, 0)
	ID3.prune(tree, validationData)
	if tree != None:
		ans = ID3.evaluate(tree, dict(a=0, b=0, c=1, d=0))
		if ans != 1:
			print("pruning test failed.")
		else:
			print("pruning test succeeded.")
	else:
		print("pruning test failed -- no tree returned.")


def testID3AndTest():
	trainData = [
		dict(a=1, b=0, c=0, Class=1),
		dict(a=1, b=1, c=0, Class=1),
		dict(a=0, b=0, c=0, Class=0),
		dict(a=0, b=1, c=0, Class=1),
	]
	testData = [
		dict(a=1, b=0, c=1, Class=1),
		dict(a=1, b=1, c=1, Class=1),
		dict(a=0, b=0, c=1, Class=0),
		dict(a=0, b=1, c=1, Class=0),
	]
	tree = ID3.ID3(trainData, 0)
	fails = 0
	if tree != None:
		acc = ID3.test(tree, trainData)
		if acc == 1.0:
			print("testing on train data succeeded.")
		else:
			print("testing on train data failed.")
			fails = fails + 1
		acc = ID3.test(tree, testData)
		if acc == 0.75:
			print("testing on test data succeeded.")
		else:
			print("testing on test data failed.")
			fails = fails + 1
		if fails > 0:
			print("Failures: ", fails)
		else:
			print("testID3AndTest succeeded.")
	else:
		print("testID3andTest failed -- no tree returned.")


# inFile - string location of the house data file
def testPruningOnHouseData(inFile):
	withPruning = []
	withoutPruning = []
	data = parse.parse(inFile)
	for i in range(100):
		random.shuffle(data)
		train = data[: len(data) // 2]
		valid = data[len(data) // 2 : 3 * len(data) // 4]
		test = data[3 * len(data) // 4 :]

		tree = ID3.ID3(train, "democrat")
		acc = ID3.test(tree, train)
		print("training accuracy: ", acc)
		acc = ID3.test(tree, valid)
		print("validation accuracy: ", acc)
		acc = ID3.test(tree, test)
		print("test accuracy: ", acc)

		ID3.prune(tree, valid)
		acc = ID3.test(tree, train)
		print("pruned tree train accuracy: ", acc)
		acc = ID3.test(tree, valid)
		print("pruned tree validation accuracy: ", acc)
		acc = ID3.test(tree, test)
		print("pruned tree test accuracy: ", acc)
		withPruning.append(acc)
		tree = ID3.ID3(train + valid, "democrat")
		acc = ID3.test(tree, test)
		print("no pruning test accuracy: ", acc)
		withoutPruning.append(acc)
	print(withPruning)
	print(withoutPruning)
	print(
		"average with pruning",
		sum(withPruning) / len(withPruning),
		" without: ",
		sum(withoutPruning) / len(withoutPruning),
	)

# trainFile - relative path of the training set
# validFile - relative path of the valid set
# testFile - relative path of test set
# default - default target value
def testPruningOnData(default, trainFile, validFile = None, testFile = None, training = None, validation = None, testing = None):
	if validFile == None or testFile == None:
		data = parse.parse(trainFile)
		train = data[: len(data) // 2]
		valid = data[len(data) // 2 : 3 * len(data) // 4]
		test = data[3 * len(data) // 4 :]
	elif trainFile != None:
		train = parse.parse(trainFile)
		valid = parse.parse(validFile)
		test = parse.parse(testFile)
	elif training != None: # just for testing purposes; this has little failsafe for people who don't do all three
		train = training
		valid = validation
		test = testing
	else:
		print("please give data to compute")
	withPruning = []
	withoutPruning = []
	train_acc = []
	valid_acc = []
	test_acc = []
	train_prune = []
	valid_prune = []


	for i in range(100):
		tree = ID3.ID3(train, default)
		acc = ID3.test(tree, train) # accuracy
		train_acc.append(acc)
		# print("training accuracy: ", acc)
		acc = ID3.test(tree, valid)
		valid_acc.append(acc)
		# print("validation accuracy: ", acc)
		acc = ID3.test(tree, test)
		test_acc.append(acc)
		# print("test accuracy: ", acc)

		ID3.prune(tree, valid)
		acc = ID3.test(tree, train)
		train_prune.append(acc)
		# print("pruned tree train accuracy: ", acc)
		acc = ID3.test(tree, valid)
		valid_prune.append(acc)
		# print("pruned tree validation accuracy: ", acc)
		acc = ID3.test(tree, test)
		# print("pruned tree test accuracy: ", acc)
		withPruning.append(acc)

		tree = ID3.ID3(train + valid, default)
		acc = ID3.test(tree, test)
		# print("no pruning test accuracy: ", acc)
		withoutPruning.append(acc)

	# print(withPruning)
	# print(withoutPruning)
	train_avg = sum(train_acc) / len(train_acc)
	valid_avg = sum(valid_acc) / len(valid_acc)
	test_avg = sum(test_acc) / len(test_acc)
	train_prune_avg = sum(train_prune) / len(train_prune)
	valid_prune_avg = sum(valid_prune) / len(valid_prune)
	withPruning_avg = sum(withPruning) / len(withPruning)
	woPruning_avg = sum(withoutPruning) / len(withoutPruning)
	print(
		"\naverage training accuracy",
		train_avg,
		"\naverage validation accuracy",
		valid_avg,
		"\naverage testing accuracy",
		test_avg,
		"\naverage training with pruning",
		train_prune_avg,
		"\naverage validation with pruning",
		valid_prune_avg,
		"\naverage with pruning",
		withPruning_avg,
		"\n without: ",
		woPruning_avg,
	)
	# return [train_avg, valid_avg, test_avg, train_prune_avg, valid_prune_avg, withPruning_avg, woPruning_avg]

testID3AndEvaluate()
testID3AndTest()
testPruning()
print("\nhouse data")
testPruningOnHouseData("HW#1/house_votes_84.data")
print("\ncar data")
testPruningOnData("unacc", "HW#1/cars_train.data", "HW#1/cars_valid.data", "HW#1/cars_test.data")

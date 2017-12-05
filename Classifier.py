import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from random import uniform

example_width = 0
example_height = 0
num_classes = 0
num_examples = 0
alpha = 1.0
bias = 0.5
epochs = 8

# Check for proper usage
if(len(sys.argv) != 5):
	print("USAGE: Classifier.py <training_labels> <training_examples> <test_labels> <test_examples>")
	sys.exit()
write = sys.stdout.write

example_width = 28
example_height = 28
num_classes = 10
value_map = {}
value_map[' '] = 0
value_map['+'] = 1
value_map['#'] = 1
weight_modes = {}
weight_modes[0] = 'zero'
weight_modes[1] = 'random'
order_modes = {}
order_modes[0] = 'in order'
order_modes[1] = 'random order'

# Function for classifying an example
def classify(example, w_vectors):

	products = np.dot(w_vectors, example.T)

	return np.argmax(products)

	# Calculate dot products
	for c in range(num_classes):
		dot_product = 0.0
		for j in range(example_height):
			for i in range(example_width):
				dot_product += example[j][i] * w_vectors[c][j][i]
		products[c] = dot_product + bias 

	# Return most likely class
	return products.index(max(products))

# Open training files
training_labels_file = open(sys.argv[1], 'r')
training_examples_file = open(sys.argv[2], 'r')
training_examples = []
training_labels = []

for num, line in enumerate(training_labels_file):
	training_labels.append(int(line))
	training_examples.append([])
	for j in range(example_height):
		row = training_examples_file.readline()
		for i in range(example_width):
			training_examples[num].append(value_map[row[i]])

# Close training files
training_examples_file.close()
training_labels_file.close()

training_examples = np.array(training_examples)

# Open test files
test_labels_file = open(sys.argv[3], 'r')
test_examples_file = open(sys.argv[4], 'r')
test_examples = []
test_labels = []

for num, line in enumerate(test_labels_file):
	test_labels.append(int(line))
	test_examples.append([])
	for j in range(example_height):
		row = test_examples_file.readline()
		for i in range(example_width):
			test_examples[num].append(value_map[row[i]])

# Close training files
test_examples_file.close()
test_labels_file.close()

test_examples = np.array(test_examples)

order_mode = 1
weight_mode = 0
bias = 0.5

print("Ordering: "+order_modes[order_mode])
index_order = [i for i in range(5000)]

print("Weight Mode: "+weight_modes[weight_mode])
# Initialize training results

print("Bias: %.1f" % bias)

if(weight_mode == 0):
	class_weight_vectors = np.zeros((10, example_width*example_height))
if(weight_mode == 1):
	class_weight_vectors = (20.0 * np.random.rand(10, example_height*example_width)) - 10.0

epoch_accuracy = [0.0 for e in range(epochs)]

for epoch in range(epochs):

	if(order_mode == 1):
		shuffle(index_order)

	alpha = 1/(epoch+1)

	training_num_per_class = [0 for c in range(num_classes)]
	training_num_examples = 0

	for index in index_order:
		# Get example label
		label = training_labels[index]
		
		classification = classify(training_examples[index], class_weight_vectors)

		# Check if correct
		if(classification != label):
			# Adjust weight vectors
			class_weight_vectors[label] += alpha * training_examples[index]
			class_weight_vectors[classification] -= alpha * training_examples[index]

	# Caclulate epoch accuracy
	for index, label in enumerate(training_labels):
				
		classification = classify(training_examples[index], class_weight_vectors)

		'''print("Example %d:" % num)
		for j in range(example_height):
			for i in range(example_width):
				write(str(x[j][i]))
			write('\n')

		print("True: %d    Guess: %d\n" % (label, classification))
		input() '''

		# Check if correct
		if(classification == label):
			epoch_accuracy[epoch] += 1
		training_num_per_class[label] += 1
		training_num_examples += 1

	#print("\tTraining:")
	#print("\tEpoch %d: %d correct out of %d; " %(epoch, epoch_accuracy[epoch], training_num_examples) + '%7.3f' % (100 * epoch_accuracy[epoch] / training_num_examples) + " %\n")

print("\tTraining Results:\n")

# Print training curve
for e in range(epochs):
	print("\tEpoch %d: %d correct out of %d; " %(e, epoch_accuracy[e], training_num_examples) + '%7.3f' % (100 * epoch_accuracy[e] / training_num_examples) + " %\n")

true_labels = []
guess_labels = []
total_correct = 0
test_num_examples = 0
test_class_num_examples = [0]*num_classes
classification_rate = [0]*num_classes

# MAP classification
for index, true_class in enumerate(test_labels):

	# Update number of examples
	test_class_num_examples[true_class] += 1
	test_num_examples += 1
	true_labels.append(true_class)

	# Calculate probability for each class
	best_class = classify(test_examples[index], class_weight_vectors)

	# Add label
	guess_labels.append(best_class)

	if(best_class == true_class):
		classification_rate[true_class] += 1
		total_correct += 1

print("Test Results:\n")

# Normalize classification rates
for c in range(num_classes):
	print("Class " + str(c) + ":")
	print(str(classification_rate[c]) + " correct out of " + str(test_class_num_examples[c]) + "; " + '%7.3f' % (100 * classification_rate[c] / test_class_num_examples[c]) + " %\n")
	classification_rate[c] /= test_class_num_examples[c]

print("\tTotal correct: ")
print('\t' + str(total_correct) + " out of " + str(test_num_examples) + "; " + str(100 * total_correct / test_num_examples) + " %\n")

# Compute confusion matrix
confusion_matrix = [[0 for c in range(num_classes)] for r in range(num_classes)]

for index, r in enumerate(true_labels):
	c = guess_labels[index]
	confusion_matrix[r][c] += 1

for r in range(num_classes):
	for c in range(num_classes):
		confusion_matrix[r][c] /= test_class_num_examples[r]

# Print confusion matrix
print("Confusion matrix: ")

sys.stdout.write('  ')
for i in range(num_classes):
	sys.stdout.write('   ' + str(i) + '   ')

for r in range(num_classes):
	sys.stdout.write('\n' + str(r) + ':')
	for c in range(num_classes):
		sys.stdout.write('%7.3f' % (100*confusion_matrix[r][c]))
print('\n')

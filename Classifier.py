import sys
import math
import matplotlib.pyplot as plt
import numpy as np

example_width = 0
example_height = 0
num_classes = 0
num_examples = 0
alpha = 1
epochs = 1

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

# Initialize training results
class_weight_vectors = [[[0.5 for i in range(example_width)] for j in range(example_height)] for c in range(num_classes)]
epoch_accuracy = [0.0 for e in range(epochs)]

# Function for classifying an example
def classify(example, w_vectors):
	x_vector = [[0.0 for i in range(example_width)] for j in range(example_height)]
	products = [0.0 for c in range(num_classes)]

	# Load x vector
	for j in range(example_height):
		row = example.readline()
		for i in range(example_width):
			char = row[i]
			x_vector[j][i] = value_map[char]

	# Calculate dot products
	for c in range(num_classes):
		dot_product = 0.0
		for j in range(example_height):
			for i in range(example_width):
				dot_product += x_vector[j][i] * w_vectors[c][j][i]
		products[c] = dot_product

	# Return most likely class
	return products.index(max(products)), x_vector

# Open training files
training_labels = open(sys.argv[1], 'r')
training_examples = open(sys.argv[2], 'r')

for epoch in range(epochs):

	training_num_per_class = [0 for c in range(num_classes)]
	training_num_examples = 0

	training_labels.seek(0)
	training_examples.seek(0)

	for line in training_labels:
		# Get example label
		label = int(line)
		
		classification, x_vector = classify(training_examples, class_weight_vectors)

		# Check if correct
		if(classification != label):
			# Adjust weight vectors
			for j in range(example_height):
				for i in range(example_width):
					class_weight_vectors[label][j][i] += alpha * x_vector[j][i]
					class_weight_vectors[classification][j][i] -= alpha * x_vector[j][i]

	# Caclulate epoch accuracy
	training_labels.seek(0)
	training_examples.seek(0)

	for num, line in enumerate(training_labels):
		# Get example label
		label = int(line)
		
		classification, x = classify(training_examples, class_weight_vectors)

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

# Close training files
training_examples.close()
training_labels.close()

print("Training Results:\n")

# Print training curve
for e in range(epochs):
	print("Epoch %d: %d correct out of %d; " %(e, epoch_accuracy[e], training_num_examples) + '%7.3f' % (100 * epoch_accuracy[e] / training_num_examples) + " %\n")

# Open test files
test_labels = open(sys.argv[3], 'r')
test_examples = open(sys.argv[4], 'r')

true_labels = []
guess_labels = []
total_correct = 0
test_num_examples = 0
test_class_num_examples = [0]*num_classes
classification_rate = [0]*num_classes

# MAP classification
for number, line in enumerate(test_labels):

	# Get actual class
	true_class = int(line)

	# Update number of examples
	test_class_num_examples[true_class] += 1
	test_num_examples += 1
	true_labels.append(true_class)

	# Variables for determining label
	best_class = -1
	highest_class_prob = -float("inf")

	# Calculate probability for each class
	best_class, x = classify(test_examples, class_weight_vectors)

	# Add label
	guess_labels.append(best_class)

	if(best_class == true_class):
		classification_rate[true_class] += 1
		total_correct += 1

# Close test files
test_examples.close()
test_labels.close()

print("Test Results:\n")

# Normalize classification rates
for c in range(num_classes):
	print("Class " + str(c) + ":")
	print(str(classification_rate[c]) + " correct out of " + str(test_class_num_examples[c]) + "; " + '%7.3f' % (100 * classification_rate[c] / test_class_num_examples[c]) + " %\n")
	classification_rate[c] /= test_class_num_examples[c]

print("Total correct: ")
print(str(total_correct) + " out of " + str(test_num_examples) + "; " + str(100 * total_correct / test_num_examples) + " %\n")

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

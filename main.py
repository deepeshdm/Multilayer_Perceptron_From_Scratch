
# ACUURACY = 0.9664
# TRAINING TIME = 16 MINUTES approx

#-----------------------------------------------------------------------

import numpy
from Neural_Network import neuralNetwork

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1
epochs = 7

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# --------------------------------------------------------------------------

# TRAINING THE NEURAL NETWORK

# load the mnist training data CSV file into a list
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for epoch in range(epochs):

    # train the neural network go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)


# --------------------------------------------------------------------------

# TESTING THE NEURAL NETWORK & FINDING ACCURACY

# load the mnist test data CSV file into a list
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# scorecard for how well the network performs, initially empty
scorecard = []

for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])

    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.predict(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)

    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to
        scorecard.append(0)
        pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
accuracy = scorecard_array.sum() / scorecard_array.size
print ("Accuracy : ", accuracy)


# --------------------------------------------------------------------------

# PREDICT A IMAGE

x = test_data_list[1]
all_values = x.split(',')
actual_label = int(all_values[0])
test_image = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

predicted_value = numpy.argmax(n.predict(test_image))
print("ACTUAL LABEL : {} , PREDICTED VALUE : {}".format(actual_label,predicted_value))
















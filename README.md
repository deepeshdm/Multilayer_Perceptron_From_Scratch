
The following is an implementation of a 3 layered artificial neural network in order to classify handwritten digits.

![84ed8a1aaf1887cdddde6afcc117fb0e](https://user-images.githubusercontent.com/63066870/115762153-c7703500-a3c0-11eb-9649-c02ec3a3a5ef.png)


Dataset used : MNIST dataset

Training set (60,000 samples) : http://www.pjreddie.com/media/files/mnist_train.csv

Test set (10,000 samples) : http://www.pjreddie.com/media/files/mnist_test.csv

The content of these records, or lines of text, is easy to understand:

● The first value is the label, that is, the actual digit that the handwriting is supposed to represent, such as a "7" or a "9". This is the answer the neural network is trying to learn to get right.

● The subsequent values, all comma separated, are the pixel values of the handwritten digit. The size of the pixel array is 28 x 28, so there are 784 values after the label.

Steps to fetch all pixel values of a 28x28 image from the dataset : 

1] Split that long text string of comma separated values into individual values, using the commas as the place to do the splitting.

2] Ignore the first value, which is the label, and take the remaining list of 28 * 28 = 784 values and turn them into an array which has a shape of 28 rows by 28 columns.

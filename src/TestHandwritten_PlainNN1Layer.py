import matplotlib
#matplotlib.use('Agg')  # Uncomment for Cloud Computing
import loadEmnist
import numpy as np
import tensorflow as tf
import random
import NN_Helpers as nnh
import matplotlib.pyplot as plt
import time

imgWidth = 28
num_classes = len(loadEmnist.enumToChar)
EPOCHS = 100
BATCH_SIZE = 64
LEARN_RATE = 0.01
num_neurons = [800]
num_classes = len(loadEmnist.enumToChar)

# Load Data
trainDat = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-train-data.npy')
trainLabels = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-train-labels.npy')
trainLabels = np.eye(num_classes,dtype=float)[trainLabels.astype(int)]
testDat = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-test-data.npy')
testLabels = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-test-labels.npy')
testLabels = np.eye(num_classes,dtype=float)[testLabels.astype(int)]

handwrittenDat = loadEmnist.loadEmnistFromNPY('../data/handwritten/binaryLetters.npy')
handwrittenLabels = loadEmnist.loadEmnistFromNPY('../data/handwritten/handwritten_labels.npy')
handwrittenLabelsOneHot = np.eye(num_classes,dtype=float)[handwrittenLabels.astype(int)]
trainDataSize = trainDat.shape[0]

# Function to generate tensorflow weight variable
def weight_var(shape):
    shape = tf.TensorShape(shape)
    initial_values = tf.random_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial_values)

# Function to generate tensorflow bias variable
def bias_var(shape):
    initial_vals = tf.zeros(shape)
    return tf.Variable(initial_vals)



layer1_size = num_neurons[0]

############### Placeholders ##############################
#  We will feed these with inputs at train/test timeinputs_ph = tf.placeholder(tf.float32, shape=[None,imgWidth*imgWidth]) # [784,]
inputs_ph = tf.placeholder(tf.float32, shape=[None,imgWidth*imgWidth]) # [784,]
targets_ph = tf.placeholder(tf.float32, shape=[None,num_classes]) # [47]

############### Tensorflow Variables ######################
# -- These are the values that the optimizer function will adjust during training
    ## Vars for First Fully Connected Layer
w1 = weight_var([inputs_ph.shape[1],layer1_size]) #[16,784]
b1 = bias_var([layer1_size])

    ## Vars for Output Layer
w2 = weight_var([layer1_size,num_classes])  #[47, 24]
b2 = bias_var([num_classes])

############### Network Structure ########################
## First Fully Connected Layer (Could try different activation functions)
a1 = tf.sigmoid(tf.add(tf.matmul(inputs_ph,w1), b1)) # [16,]

## Second Fully Connected Layer (Could try using activation functions)
outputs = tf.add(tf.matmul(a1,w2), b2)

############### Tensorflow Operations ####################
## We will use these to evaluate and train the neural network
# Loss function to minimize
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets_ph, logits=outputs))

# Prediction: Takes argmax of output layer to choose the most probable classification of the input
prediction = tf.argmax(outputs,1)

# Percent of predictions correct
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,tf.argmax(targets_ph,1)),dtype=tf.float32))

# Optimizer to minimize loss. Could try different optimizer as well as varying the learning rate
optimizer = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss, var_list=[w1,b1,w2,b2])

# Model Saver
saver = tf.train.Saver()

# Run test
with tf.Session() as sess:
    saver.restore(sess, ("model/model_plain_" + str(layer1_size) + "Neur_1Layer.ckpt"))
    testAcc = sess.run([accuracy], feed_dict={inputs_ph: testDat, targets_ph: testLabels})
    l,pred,a = sess.run([loss,prediction,accuracy], feed_dict={inputs_ph: handwrittenDat, targets_ph: handwrittenLabelsOneHot})

for i in range(handwrittenDat.shape[0]):
    target = handwrittenLabels[i]
    p = pred[i]
    print('\nTrue character: ' + loadEmnist.enumToChar[target] + ' is displayed below.   NN Prediction: ' + loadEmnist.enumToChar[p])
    loadEmnist.printImg(handwrittenDat[i],28,0.25)

print('\nTestdat accuracy' + str(testAcc))
print('\nHandwritten accuracy: ' + str(a))


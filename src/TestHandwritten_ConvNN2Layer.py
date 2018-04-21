import matplotlib
matplotlib.use('Agg') 
import loadEmnist
import numpy as np
import tensorflow as tf
import random
import NN_Helpers as nnh
import matplotlib.pyplot as plt

imgWidth = 28
num_classes = len(loadEmnist.enumToChar)
LEARN_RATE = 0.0001
FILTER_SIZE = 5
NUM_FILTERS = 32
FILTER_SIZE1 = 5
FILTER_SIZE2 = 5
NUM_FILTERS1 = 32
NUM_FILTERS2 = 64
DROP_RATE = 0.5
SMOOTHING_WINDOW = 25
master_accuracy_lst = []
num_neurons = [40]
testDat = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-test-data.npy')
#testDat = testDat*255
testDat = testDat.astype(int)
testDat = testDat.reshape([testDat.shape[0],imgWidth,imgWidth,1])
    #testDat = loadEmnist.loadEmnistFromNPY('../data/MNIST/MNIST-test-data.npy')
testLabels = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-test-labels.npy')
    #testLabels = loadEmnist.loadEmnistFromNPY('../data/MNIST/MNIST-test-labels.npy')
testLabels = np.eye(num_classes,dtype=float)[testLabels.astype(int)]

handwrittenDat = loadEmnist.loadEmnistFromNPY('../data/handwritten/binaryLetters.npy')
handwrittenDat = handwrittenDat * 255
handwrittenDat = handwrittenDat.astype(int)
handwrittenDat = handwrittenDat.reshape([handwrittenDat.shape[0],imgWidth,imgWidth,1])
handwrittenLabels = loadEmnist.loadEmnistFromNPY('../data/handwritten/handwritten_labels.npy')
handwrittenLabelsOneHot = np.eye(num_classes,dtype=float)[handwrittenLabels.astype(int)]

def weight_var(shape):
    shape = tf.TensorShape(shape)
    initial_values = tf.truncated_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial_values)

def bias_var(shape):
    initial_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(initial_vals)

def conv_2d_layer(x,w,b,activationFn):
    layer = tf.nn.conv2d(x,w,strides=[1,1,1,1], padding='SAME')
    if activationFn == 'relu':
      return tf.nn.relu(tf.add(layer,b))
    else:
      return tf.add(layer,b)

def max_pool_NxN(x,reduction_factor):
    return tf.nn.max_pool(x, ksize=[1,reduction_factor,reduction_factor,1], strides=[1,reduction_factor,reduction_factor,1],padding='SAME')



layer1_size = num_neurons[0]
layer2_size = layer1_size
# Placeholders (placeholders that we will feed with values during training)
inputs_ph = tf.placeholder(tf.float32, shape=[None,imgWidth,imgWidth,1]) # [784,]
targets_ph = tf.placeholder(tf.float32, shape=[None,num_classes]) # [47]
retain_prob = tf.placeholder(tf.float32)

# Variables (weights that will be adjusted by the minimize() function)
## Convolutional layer 1
conv_w1 = weight_var([FILTER_SIZE1,FILTER_SIZE1,1,NUM_FILTERS1])
conv_b1 = bias_var([NUM_FILTERS1])
## Convolutional layer 2
conv_w2 = weight_var([FILTER_SIZE2, FILTER_SIZE2,NUM_FILTERS1,NUM_FILTERS2])
conv_b2 = bias_var([NUM_FILTERS2])
## First Fully Connected 
w1 = weight_var([7*7*NUM_FILTERS2,layer1_size]) #[16,784]
b1 = bias_var([layer1_size])
## Second Fully Connected Layer
w2 = weight_var([layer1_size,layer2_size])  #[24,16]
b2 = bias_var([layer2_size])
## Output Layer
w3 = weight_var([layer1_size,num_classes])  #[47, 24]
b3 = bias_var([num_classes])

# Network Structure
## First Convolutional Layer
conv_layer_1 = conv_2d_layer(inputs_ph,conv_w1,conv_b1,'relu')
max_pool_1 = max_pool_NxN(conv_layer_1,2)
## Second Convolutional Layer
conv_layer_2 = conv_2d_layer(max_pool_1,conv_w2,conv_b2,'relu')
max_pool_2 = max_pool_NxN(conv_layer_2,2)

dropout = tf.nn.dropout(max_pool_2, retain_prob)

### Convert to vector
FCL_Input = tf.reshape(dropout, [-1,7*7*NUM_FILTERS2])
## First Fully Connected Layer (Could try different activation functions)
a1 = tf.nn.relu(tf.add(tf.matmul(FCL_Input,w1), b1)) # [16,]
## Second Fully Connected Layer (Could try different activation functions)
a2 = tf.nn.relu(tf.add(tf.matmul(a1,w2), b2)) # [24,]

## Output Layer (Could try adding activation function)
#outputs = tf.sigmoid(tf.matmul(a2,w3) + b3) # [47,]
outputs = tf.add(tf.matmul(a2,w3), b3)

#Loss to minimize (could try different loss function)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets_ph, logits=outputs))
#loss = tf.losses.mean_squared_error(outputs, targets_ph)

prediction = tf.argmax(outputs,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,tf.argmax(targets_ph,1)),dtype=tf.float32))

# Optimizer to minimize loss. Could try different optimizer as well as varying the learning rate
optimizer = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss, var_list=[conv_w1, conv_b1, conv_w2, conv_b2,w1,b1,w2,b2,w3,b3])

# Model Saver
saver = tf.train.Saver()

    ############### Training / Validation Loop ################
with tf.Session() as sess:
    saver.restore(sess,("model/model_conv_" + str(layer1_size) + "Neur_2Layer.ckpt"))
    testAcc = sess.run([accuracy], feed_dict={inputs_ph: testDat, targets_ph: testLabels, retain_prob: 1.0})
    l,pred,a = sess.run([loss,prediction,accuracy], feed_dict={inputs_ph: handwrittenDat, targets_ph: handwrittenLabelsOneHot, retain_prob: 1.0})

for i in range(handwrittenDat.shape[0]):
    target = handwrittenLabels[i]
    p = pred[i]
    print('\nTrue character: ' + loadEmnist.enumToChar[target] + ' is displayed below.   NN Prediction: ' + loadEmnist.enumToChar[p])
    loadEmnist.printImg(handwrittenDat[i].reshape([28*28]),28,0.25)

print('\nTestdat accuracy' + str(testAcc))
print('\nHandwritten accuracy: ' + str(a))
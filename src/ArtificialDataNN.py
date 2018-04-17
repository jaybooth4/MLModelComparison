import numpy as np
import tensorflow as tf
import random
import generate_data as gd
import matplotlib.pyplot as plt

imgWidth = 28
layer1_size = 1
#layer2_size = 100
num_classes = 2
EPOCHS = 2000
BATCH_SIZE = 64
LEARN_RATE = 0.01
nDims = 2

inmeans = [[0,0],[2,2],[0,2]]
incovs = [[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]]
datasize = 1000

trainDat, testDat, trainLabels, testLabels = gd.generate_data(inmeans, incovs, datasize)
trainLabels = trainLabels - 1
testLabels = testLabels - 1
trainLabels = np.eye(num_classes)[trainLabels.astype(int)]
testLabels = np.eye(num_classes)[testLabels.astype(int)]


trainDataSize = trainDat.shape[0]
def weight_var(shape):
    shape = tf.TensorShape(shape)
    initial_values = tf.random_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial_values)

def bias_var(shape):
    initial_vals = tf.zeros(shape)
    return tf.Variable(initial_vals)



# Placeholders
inputs_ph = tf.placeholder(tf.float32, shape=[None,nDims]) # [784,]
targets_ph = tf.placeholder(tf.float32, shape=[None,num_classes]) # [47]

# Variables
## First Fully Connected Layer
w1 = weight_var([inputs_ph.shape[1],layer1_size]) #[16,784]
b1 = bias_var([layer1_size])
## Second Fully Connected Layer
#w2 = weight_var([layer1_size,layer2_size])  #[24,16]
#b2 = bias_var([layer2_size])
## Output Layer
w3 = weight_var([layer1_size,num_classes])  #[47, 24]
b3 = bias_var([num_classes])

# Network Structure
## First Fully Connected Layer (Could try different activation functions)
a1 = tf.sigmoid(tf.add(tf.matmul(inputs_ph,w1), b1)) # [16,]
## Second Fully Connected Layer (Could try different activation functions)
#a2 = tf.sigmoid(tf.add(tf.matmul(a1,w2), b2)) # [24,]
## Output Layer (Could try adding activation function)
#outputs = tf.sigmoid(tf.matmul(a2,w3) + b3) # [47,]
outputs = tf.add(tf.matmul(a1,w3), b3)

#Loss to minimize (could try different loss function)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets_ph, logits=outputs))
#loss = tf.losses.mean_squared_error(outputs, targets_ph)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs,1),tf.argmax(targets_ph,1)),dtype=tf.float32))

# Optimizer to minimize loss. Could try different optimizer as well as varying the learning rate
optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss, var_list=[w1,b1,w3,b3])

# saver to save and later restore state (don't know how to restore yet)
saver = tf.train.Saver()

# Saves event logs
acc_report = tf.summary.scalar('accuracy',accuracy)

writer = tf.summary.FileWriter('./log')
writer.add_graph(tf.get_default_graph())


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        print('EPOCH ' + str(i))
        index = random.sample(range(trainDataSize),k=trainDataSize)
        for j in range(0,trainDataSize,BATCH_SIZE):
            _,acc = sess.run([optimizer,accuracy], feed_dict={inputs_ph: trainDat[index[j:min(j+BATCH_SIZE,trainDataSize-1)]],   targets_ph: trainLabels[index[j:min(j+BATCH_SIZE,trainDataSize-1)]]} ) 
            print(str(acc))
        acc_rep,l,a = sess.run([acc_report,loss,accuracy], feed_dict={inputs_ph: testDat, targets_ph: testLabels})
        writer.add_summary(acc_rep,i)
        print(l)
        print(a)
        save_path = saver.save(sess, "./model/model.ckpt")
        print("Model saved in path: %s" % save_path)



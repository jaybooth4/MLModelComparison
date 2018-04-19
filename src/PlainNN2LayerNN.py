import loadEmnist
import numpy as np
import tensorflow as tf
import random
import NN_Helpers as nnh
import matplotlib.pyplot as plt

imgWidth = 28
#layer1_size = 10
num_classes = len(loadEmnist.enumToChar)
print(str(num_classes) + ' classes')
EPOCHS = 1000
BATCH_SIZE = 64
LEARN_RATE = 0.0001
SMOOTHING_WINDOW = 10 # Number of previous epoch accuracies to consider when deciding whether to stop
master_accuracy_lst = [] # This will store the accuracy at each epoch
num_neurons = [100, 200, 400, 800, 1600]# [50, 100, 200, 400, 800]

# Load Data
trainDat = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-train-data.npy')
trainLabels = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-train-labels.npy')
trainLabels = np.eye(num_classes,dtype=float)[trainLabels.astype(int)]
testDat = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-test-data.npy')
testLabels = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-test-labels.npy')
testLabels = np.eye(num_classes,dtype=float)[testLabels.astype(int)]

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


for iteration in range(len(num_neurons)):
    accuracy_lst = [] # this will store the accuracy at each epoch

    layer1_size = num_neurons[iteration]

    ############### Placeholders ##############################
    #  We will feed these with inputs at train/test timeinputs_ph = tf.placeholder(tf.float32, shape=[None,imgWidth*imgWidth]) # [784,]
    inputs_ph = tf.placeholder(tf.float32, shape=[None,imgWidth*imgWidth])
    targets_ph = tf.placeholder(tf.float32, shape=[None,num_classes])

    ############### Tensorflow Variables ######################
    # -- These are the values that the optimizer function will adjust during training
        ## Vars for First Fully Connected Layer
    w1 = weight_var([inputs_ph.shape[1],layer1_size])
    b1 = bias_var([layer1_size])

        ## Vars for Second Fully Connected Layer
    w2 = weight_var([layer1_size,layer1_size])
    b2 = bias_var([layer1_size])

        ## Vars for Output Layer
    w3 = weight_var([layer1_size,num_classes])
    b3 = bias_var([num_classes])

    ############### Network Structure ########################
    ## First Fully Connected Layer (Could try different activation functions)
    a1 = tf.sigmoid(tf.add(tf.matmul(inputs_ph,w1), b1)) 

    ############### Network Structure ########################
    ## Second Fully Connected Layer (Could try different activation functions)
    a2 = tf.sigmoid(tf.add(tf.matmul(a1,w2), b2)) 

    ## Second Fully Connected Layer (Could try using activation functions)
    outputs = tf.add(tf.matmul(a2,w3), b3)

    ############### Tensorflow Operations ####################
    ## We will use these to evaluate and train the neural network
    # Loss function to minimize
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets_ph, logits=outputs))

    # Prediction: Takes argmax of output layer to choose the most probable classification of the input
    prediction = tf.argmax(outputs,1)

    # Percent of predictions correct
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,tf.argmax(targets_ph,1)),dtype=tf.float32))

    # Optimizer to minimize loss. Could try different optimizer as well as varying the learning rate
    optimizer = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss, var_list=[w1,b1,w2,b2,w3,b3])

    ############### Training / Validation Loop ################
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCHS):
            index = random.sample(range(trainDataSize),k=trainDataSize)
            for j in range(0,trainDataSize,BATCH_SIZE):
                _,acc = sess.run([optimizer,accuracy], feed_dict={inputs_ph: trainDat[index[j:min(j+BATCH_SIZE,trainDataSize-1)]],   targets_ph: trainLabels[index[j:min(j+BATCH_SIZE,trainDataSize-1)]]} ) 
            l,a = sess.run([loss,accuracy], feed_dict={inputs_ph: testDat, targets_ph: testLabels})
            print('EPOCH ' + str(i) + ": "+ str(a))
            accuracy_lst.append(acc)
            if (i > 5*SMOOTHING_WINDOW):
                if nnh.finished_training(accuracy_lst,SMOOTHING_WINDOW) == True:
                    break
    master_accuracy_lst.append((layer1_size,accuracy_lst))
    print(str(layer1_size) + " Neurons: Final Accuracy after " + str(len(accuracy_lst)) + " Epochs:" + str(a))

for lst in master_accuracy_lst:
    plt.plot(range(0,len(lst[1])), lst[1], label=(str(lst[0]) + 'Neuron'))
    print(str(lst[0]) + ' Neurons: Final Accuracy after ' + str(len(lst[1])) + ' Epochs: ' + str(lst[1][len(lst[1])-1]))
    with open('log.out','a') as logfile:
        logfile.write(str(lst[0]) + ' Neurons: Final Accuracy after ' + str(len(lst[1])) + ' Epochs: ' + str(lst[1][len(lst[1])-1]) + '\n')
plt.xlabel('Epoch #')
plt.ylabel('Accuracy %')
plt.title('Training Curves')
plt.legend(loc='lower right')
plt.savefig('train_plot.png')
plt.show()
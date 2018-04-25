import matplotlib
#matplotlib.use('Agg')  # Uncomment for Cloud Computing
import loadEmnist
import numpy as np
import tensorflow as tf
import random
import NN_Helpers as nnh
import matplotlib.pyplot as plt
import time
from spamTextParser import bagOfWordsParser

imgWidth = 28
EPOCHS = 1000
BATCH_SIZE = 64
LEARN_RATE = 0.0001
SMOOTHING_WINDOW = 15 # Number of previous epoch accuracies to consider when deciding whether to stop
master_accuracy_lst = [] # This will store the accuracy at each epoch
num_neurons = [100, 200, 400, 800]

dataSet = "EMNIST"
#dataSet = "SPAM"

# Load Data
if dataSet == "EMNIST":
    num_classes = len(loadEmnist.enumToChar)
    print(str(num_classes) + ' classes')
    trainDat = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-train-data.npy')
    trainLabels = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-train-labels.npy')
    trainLabels = np.eye(num_classes,dtype=float)[trainLabels.astype(int)]
    testDat = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-test-data.npy')
    testLabels = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-test-labels.npy')
    testLabels = np.eye(num_classes,dtype=float)[testLabels.astype(int)]
elif dataSet == "SPAM":
    # Read in SPAM dataset
    num_classes = 2
    trainLabels, trainDat, testLabels, testDat = \
             bagOfWordsParser('../data/SPAM/SMSSpamCollection', 0.2)
    trainLabels = np.eye(num_classes,dtype=float)[trainLabels.astype(int)]
    testLabels = np.eye(num_classes,dtype=float)[testLabels.astype(int)]

print("Shape of training labels: ", trainLabels.shape)
print("Shape of training data: ", trainDat.shape)
print("Shape of testing labels: ", testLabels.shape)
print("Shape of testing data: ", testDat.shape)
print("Total number of samples: ", trainDat.shape[1])
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

# Test all the layer sizes in num_neurons
for iteration in range(len(num_neurons)):
    tf.reset_default_graph()
    
    train_start = time.time()
    accuracy_lst = [] # this will store the accuracy at each epoch

    layer1_size = num_neurons[iteration]

    ## Log current iteration
    with open('2L_Plain_log.out','a') as logfile:
        logfile.write('\n1 Layer of '+str(layer1_size) + ' Neurons. Learn Rate: ' + str(LEARN_RATE))
    print('\n1 Layer of '+str(layer1_size) + ' Neurons. Learn Rate: ' + str(LEARN_RATE))
    ############### Placeholders ##############################
    #  We will feed these with inputs at train/test timeinputs_ph = tf.placeholder(tf.float32, shape=[None,imgWidth*imgWidth]) # [784,]
    inputs_ph = tf.placeholder(tf.float32, shape=[None,trainDat.shape[1]])
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

    # Model Saver
    saver = tf.train.Saver()

    ############### Training / Validation Loop ################
    with tf.Session() as sess:
        print("Starting Session")
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCHS):
            index = random.sample(range(trainDataSize),k=trainDataSize)
            for j in range(0,trainDataSize,BATCH_SIZE):
                _,acc = sess.run([optimizer,accuracy], feed_dict={inputs_ph: trainDat[index[j:min(j+BATCH_SIZE,trainDataSize-1)]],   targets_ph: trainLabels[index[j:min(j+BATCH_SIZE,trainDataSize-1)]]} ) 
            l,a = sess.run([loss,accuracy], feed_dict={inputs_ph: testDat, targets_ph: testLabels})
            print('EPOCH ' + str(i) + ": "+ str(a))
            accuracy_lst.append(a)
            with open('2L_Plain_log.out','a') as logfile:
                logfile.write('\nEpoch ' + str(i) + ': Loss: ' + str(l) + ' Accuracy: ' + str(a))
            if (i > 5*SMOOTHING_WINDOW):
                if nnh.finished_training(accuracy_lst,SMOOTHING_WINDOW) == True:
                    break
        save_path = saver.save(sess, ("model/model_plain_" + str(layer1_size) + "Neur_2Layer.ckpt"))

    train_end = time.time()
    train_time = train_end - train_start
    master_accuracy_lst.append((layer1_size,accuracy_lst,train_time))
    print(str(layer1_size) + " Neurons: Final Accuracy after " + str(len(accuracy_lst)) + " Epochs:" + str(a))

    ############### Generate Plots ################
for lst in master_accuracy_lst:
    plt.plot(range(0,len(lst[1])), lst[1], label=(str(lst[0]) + 'Neuron'))
    print('\n' + str(lst[0]) + ' Neurons: Final Accuracy after ' + str(len(lst[1])) + ' Epochs: ' + str(lst[1][len(lst[1])-1]))
    with open('2L_Plain_log.out','a') as logfile:
        logfile.write('\n' + str(lst[0]) + ' Neurons: Final Accuracy after ' + str(len(lst[1])) + ' Epochs: ' + str(lst[1][len(lst[1])-1])  + '.  Train Time: ' + str(lst[2]))
plt.xlabel('Epoch #')
plt.ylabel('Accuracy %')
plt.title('Training Curves')
plt.legend(loc='lower right')
plt.savefig('train_plot_2L.png')
plt.show()
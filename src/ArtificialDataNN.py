import numpy as np
import tensorflow as tf
import random
import generate_data as gd
import NN_Helpers as nnh
import matplotlib.pyplot as plt

imgWidth = 28
#layer1_size = 1
EPOCHS = 20000 # Number of times to train on the entire sample data
BATCH_SIZE = 64 # Number of samples to process for each optimization step
LEARN_RATE = 0.0005 # Learning rate parameter for the optimization function
SMOOTHING_WINDOW = 200 # Number of previous epoch accuracies to consider when deciding whether to stop
master_accuracy_lst = [] # This will store the accuracy at each epoch for all iterations
num_neurons = [1,11,21,31,41,51] # We will test the NN with each of these number of Hidden Layer Neurons

# Set artificial data parameters
inmeans = [[0,0,0],[0,0,1]]
incovs = [[[1,0,0],[0,1,0], [0,0,1]],   [[1,0,0],[0,1,0],[0,0,1]]]
datasize = 1000

# Generate Data
trainDat, testDat, trainLabels, testLabels = gd.generate_data(inmeans, incovs, datasize)

# Get dimensionality info from training data
num_classes = int(max(trainLabels) + 1)
nDims = trainDat[0].size
trainDataSize = trainDat.shape[0]

# Convert labels to one-hot (ex: '3' -> [0 0 0 1 0 0 0 0 0])
trainLabels = np.eye(num_classes)[trainLabels.astype(int)]
testLabels = np.eye(num_classes)[testLabels.astype(int)]

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
    accuracy_lst = [] # This will store the accuracy at each epoch
    train_accuracy_lst = []
    layer1_size = num_neurons[iteration]
    ############### Placeholders ##############################
    #  We will feed these with inputs at train/test time
    inputs_ph = tf.placeholder(tf.float32, shape=[None,nDims]) # [784,]
    targets_ph = tf.placeholder(tf.float32, shape=[None,num_classes]) # [47]

    ############### Tensorflow Variables ######################
    # -- These are the values that the optimizer function will adjust during training
        ## Vars for First Fully Connected Layer
    w1 = weight_var([inputs_ph.shape[1],layer1_size])
    b1 = bias_var([layer1_size])
        ## Vars for second hidden layer
    w2 = weight_var([layer1_size,layer1_size])
    b2 = bias_var([layer1_size])
        ## Vars for output layer
    w3 = weight_var([layer1_size,num_classes])
    b3 = bias_var([num_classes])
    ############### Network Structure ########################
    ## First Fully Connected Layer (Could try different activation functions)
    a1 = tf.sigmoid(tf.add(tf.matmul(inputs_ph,w1), b1))

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
    optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss, var_list=[w1,b1,w2,b2,w3,b3])

    ############### Training / Validation Loop ################

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCHS):
            index = random.sample(range(trainDataSize),k=trainDataSize)
            for j in range(0,trainDataSize,BATCH_SIZE):
                _,acc = sess.run([optimizer,accuracy], feed_dict={inputs_ph: trainDat[index[j:min(j+BATCH_SIZE,trainDataSize-1)]],   targets_ph: trainLabels[index[j:min(j+BATCH_SIZE,trainDataSize-1)]]} ) 
            l,a = sess.run([loss,accuracy], feed_dict={inputs_ph: testDat, targets_ph: testLabels})
            #with open('log.out','a') as logfile:
             #   logfile.write("Epoch "+ str(i) + ": "+ str(a)+"\n")
            print('EPOCH ' + str(i) + ": " + str(a))
            accuracy_lst.append(a)
            if (i > 5*SMOOTHING_WINDOW):
                if nnh.finished_training(accuracy_lst,SMOOTHING_WINDOW) == True:
                    break
    master_accuracy_lst.append((layer1_size,accuracy_lst))
    print(str(layer1_size) + " Neurons: Final Accuracy after " + str(len(accuracy_lst)) + " Epochs:" + str(a))

for lst in master_accuracy_lst:
    plt.plot(range(0,len(lst[1])), lst[1], label=(str(lst[0]) + 'Neuron'))
    print(str(lst[0]) + ' Neurons: Final Accuracy after ' + str(len(lst[1])) + ' Epochs: ' + str(lst[1][len(lst[1])-1]))
    with open('artificial/log.out','a') as logfile:
        logfile.write(str(lst[0]) + ' Neurons: Final Accuracy after ' + str(len(lst[1])) + ' Epochs: ' + str(lst[1][len(lst[1])-1]) + '\n')

plt.xlabel('Epoch #')
plt.ylabel('Accuracy %')
plt.title('Training Curves')
plt.legend(loc='lower right')
plt.savefig('train_plot.png')
plt.show()
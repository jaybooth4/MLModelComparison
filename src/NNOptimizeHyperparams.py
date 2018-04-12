import loadEmnist
import numpy as np
import tensorflow as tf
import random

imgWidth = 28

num_classes = len(loadEmnist.enumToChar)
#num_classes = 10
EPOCHS = 100
BATCH_SIZE = 64
LEARN_RATE = 0.2

trainDat = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-train-data.npy')
    #trainDat = loadEmnist.loadEmnistFromNPY('../data/MNIST/MNIST-train-data.npy')
trainLabels = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-train-labels.npy')
    #trainLabels = loadEmnist.loadEmnistFromNPY('../data/MNIST/MNIST-train-labels.npy')
trainLabels = np.eye(num_classes,dtype=float)[trainLabels.astype(int)]

testDat = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-test-data.npy')
    #testDat = loadEmnist.loadEmnistFromNPY('../data/MNIST/MNIST-test-data.npy')
testLabels = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-test-labels.npy')
    #testLabels = loadEmnist.loadEmnistFromNPY('../data/MNIST/MNIST-test-labels.npy')
testLabels = np.eye(num_classes,dtype=float)[testLabels.astype(int)]

trainDataSize = trainDat.shape[0]
def weight_var(shape):
    shape = tf.TensorShape(shape)
    initial_values = tf.random_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial_values)

def bias_var(shape):
    initial_vals = tf.zeros(shape)
    return tf.Variable(initial_vals)


for size in range(30,300,20):
    layer1_size = size
    layer2_size = size
    for restart in range(30):
        print("*********** RESTART " + str(restart) +".  LAYERSIZE "+ str(size) + "***********")
        # Placeholders
        inputs_ph = tf.placeholder(tf.float32, shape=[None,imgWidth*imgWidth]) # [784,]
        targets_ph = tf.placeholder(tf.float32, shape=[None,num_classes]) # [47]

        # Variables
        ## First Fully Connected Layer
        w1 = weight_var([inputs_ph.shape[1],layer1_size]) #[16,784]
        b1 = bias_var([layer1_size])
        ## Second Fully Connected Layer
        w2 = weight_var([layer1_size,layer2_size])  #[24,16]
        b2 = bias_var([layer2_size])
        ## Output Layer
        w3 = weight_var([layer2_size,num_classes])  #[47, 24]
        b3 = bias_var([num_classes])

        # Network Structure
        ## First Fully Connected Layer
        a1 = tf.sigmoid(tf.add(tf.matmul(inputs_ph,w1), b1)) # [16,]
        ## Second Fully Connected Layer
        a2 = tf.sigmoid(tf.add(tf.matmul(a1,w2), b2)) # [24,]
        ## Output Layer
        #outputs = tf.sigmoid(tf.matmul(a2,w3) + b3) # [47,]
        outputs = tf.add(tf.matmul(a2,w3), b3)
        output = tf.argmax(outputs,1)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets_ph, logits=outputs))
        #loss = tf.losses.mean_squared_error(outputs, targets_ph)
        #loss = tf.reduce_mean(
        #    tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=tf.one_hot(targets_ph,num_classes))
        #)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs,1),tf.argmax(targets_ph,1)),dtype=tf.float32))

        optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss, var_list=[w1,b1,w3,b3])

        # saver to save and later restore state
        saver = tf.train.Saver()

        writer = tf.summary.FileWriter('.')
        writer.add_graph(tf.get_default_graph())

        test = 0

        if test ==1:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for i in range(0,5000):
                    sess.run(optimizer, feed_dict={inputs_ph: trainDat[0:BATCH_SIZE], targets_ph: trainLabels[0:BATCH_SIZE]})
                    a,b,c = sess.run([outputs,output,accuracy], feed_dict={inputs_ph: trainDat[0:100], targets_ph: trainLabels[0:100]})
                    print("\ni="+str(i))
                    #print(a)
                    print(b)
                    print(c)

        else:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(EPOCHS):
                    #print("trainDataSize = " + str(trainDataSize))
                    print('EPOCH ' + str(i))
                    index = random.sample(range(trainDataSize),k=trainDataSize)
                    tot = 0
                    for j in range(0,trainDataSize,BATCH_SIZE):
                    #for j in range(0,100*BATCH_SIZE,BATCH_SIZE):
                        _,acc = sess.run([optimizer,accuracy], feed_dict={inputs_ph: trainDat[index[j:min(j+BATCH_SIZE,trainDataSize-1)]],   targets_ph: trainLabels[index[j:min(j+BATCH_SIZE,trainDataSize-1)]]} ) 
                        tot += min(j+BATCH_SIZE,trainDataSize-1) - j + 1
                        #print(str(j) + " Acc = " + str(acc))
                    # if j % 640 == 0:
                            #print(str(acc))
                    #print("Images in Epoch: "+str(tot))
                    print(sess.run([loss,accuracy], feed_dict={inputs_ph: testDat, targets_ph: testLabels}))
                    save_path = saver.save(sess, "./model/model.ckpt")
                    print("Model saved in path: %s" % save_path)



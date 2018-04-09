import loadEmnist
import numpy as np


# Read in EMNIST test dataset from CSV
# testDat: np.ndarray[18800][28*28]
# testLabels: np.ndarray[18800]
#testDat, testLabels = loadEmnist.loadEmnistDataset('../data/EMNIST/emnist-balanced-test.csv',18800,28)

# Read in EMNIST train data from .npy
testDat = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-train-data.npy')
testLabels = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-train-data.npy')

# Read in EMNIST train dataset from CSV
# testDat: np.ndarray[112800][28*28]
# testLabels: np.ndarray[112800]
# trainDat, trainLabels = loadEmnist.loadEmnistDataset('../data/EMNIST/emnist-balanced-train.csv', 112800,28)

# Read in EMNIST test data from .npy
testDat = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-test-data.npy')
testLabels = loadEmnist.loadEmnistFromNPY('../data/EMNIST/balanced-test-labels.npy')

# Print first 10 of test dataset digits to terminal
for i in range(0,10):
    print('Character: ' + loadEmnist.enumToChar[testLabels[i]])
    loadEmnist.printImg(testDat[i][:],28,128)
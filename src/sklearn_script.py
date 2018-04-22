
# coding: utf-8

# In[6]:

import csv
import numpy as np
import os, zipfile

enumToChar = {0:'0',1:'1',2:'2',3 :'3',4 :'4',5 :'5',6 :'6',7 :'7',8 :'8',9 :'9',10: 'A',11: 'B',12: 'C',13: 'D',14: 'E',15: 'F',16: 'G',17: 'H',18 :'I',19 :'J',20 :'K',
21 :'L',22 :'M',23 :'N',24 :'O',25 :'P',26 :'Q',27 :'R',28 :'S',29 :'T',30 :'U',31 :'V',32 :'W',33 :'X',34 :'Y',35 :'Z',36 :'a',37 :'b',38 :'d',39 :'e',40 :'f',41 :'g',42 :'h',43 :'n',44 :'q',45 :'r',46 :'t'
}

def loadEmnistDatasetFromCSV(filename, imageCount, rowLength):
    data = np.ndarray(shape=(imageCount,rowLength*rowLength), dtype = int)
    labels = np.ndarray(imageCount)
    with open(filename, 'r') as inFile:
        csvReader = csv.reader(inFile)
        imageNum = -1
        for row in csvReader:
            imageNum += 1
            print("imageNum = "+str(imageNum))
            i = 0
            for p in row:
                if i == 0:
                    labels[imageNum] = p
                else:
                    t = i-1
                    idx =  int(t/rowLength) + rowLength * (t % rowLength)
                    data[imageNum][idx] = p
                i += 1
    return data,labels

def loadEmnistFromNPY(filename):
    try:
        ret = np.load(filename)
    except FileNotFoundError:
        zipRef = zipfile.ZipFile('../data/EMNIST/balanced-data.zip')
        zipRef.extractall('../data/EMNIST')
        zipRef.close()
        ret = np.load(filename)

    return ret

def printImg(data,rowSize,thresh):
    length = max(data.shape)
    if(length == rowSize*rowSize):
        i = 0
    elif(length == rowSize*rowSize + 1):
        print("Label key: " + str(data[0]))
        i = 1
    else:
        print("Invalid data. Cannot print")
        return

    render = ''
    for row in range(0,rowSize):
        for col in range(0,rowSize):
            p = data[i]
            i += 1
            if(p>thresh):
                render += 'X'
            else:
                render += ' '
        render += '\n'
    render += '\n'
    print(render)


# In[17]:


# Read in EMNIST test dataset from CSV
# testDat: np.ndarray[18800][28*28]
# testLabels: np.ndarray[18800]
#testDat, testLabels = loadEmnist.loadEmnistDataset('../data/EMNIST/emnist-balanced-test.csv',18800,28)

# Read in EMNIST train data from .npy
EMtrainDat = loadEmnistFromNPY('../data/EMNIST/balanced-train-data.npy')
EMtrainLabels = loadEmnistFromNPY('../data/EMNIST/balanced-train-labels.npy')

# Read in EMNIST train dataset from CSV
# testDat: np.ndarray[112800][28*28]
# testLabels: np.ndarray[112800]
# trainDat, trainLabels = loadEmnist.loadEmnistDataset('../data/EMNIST/emnist-balanced-train.csv', 112800,28)

# Read in EMNIST test data from .npy
EMtestDat = loadEmnistFromNPY('../data/EMNIST/balanced-test-data.npy')
EMtestLabels = loadEmnistFromNPY('../data/EMNIST/balanced-test-labels.npy')


RSelect=np.random.choice(112800,100)
SampleData = np.zeros((100,784))
count =0
for index in RSelect:
    SampleData[count]=EMtrainDat[index]
    count = count+1



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import neural_network
import time
import matplotlib.pyplot as plt
from pympler import asizeof
import numpy as np
np.random.seed(1)

save_file = "sklearn_results.txt"
# In[7]:


def fitModel(model, trainingData, trainingLabels):
    startMemory = asizeof.asizeof(model)
    startTime = time.time()
    model.fit(trainingData, trainingLabels)
    endTime = time.time()
    endMemory = asizeof.asizeof(model)
    print("Time to fit: " + str(endTime - startTime))
    print("Memory growth fit: " + str(endMemory - startMemory))
    with open(save_file,'a') as f:
        f.write('\n\tTime to fit: ' + str(endTime - startTime))
        f.write('\n\tMemory growth fit: ' + str(endMemory - startMemory))


# In[8]:


def predictModel(model, testData):
    startTime = time.time()
    predictions = model.predict(testData)
    endTime = time.time()
    print("Time to predict " + str(endTime - startTime))
    return predictions


# In[9]:


def runOnModelAndData(_model, dataTrain, dataTest, labelsTrain, labelsTest):
    model = _model()
    with open(save_file, 'a') as f:
        f.write('\nResults for ' + str(_model.__name__) + ':')
    fitModel(model, dataTrain, labelsTrain)
    predictions = predictModel(model, dataTest)
    accuracy = accuracy_score(labelsTest.tolist(), predictions)
    print(accuracy)
    with open(save_file, 'a') as f:
        f.write('\n\tAccuracy: ' + str(accuracy))
    #plotBoundary(model, dataTrain, labelsTrain)


# In[10]:


def KNNModel():
    return neighbors.KNeighborsClassifier(n_jobs=-1)
    #return neighbors.KNeighborsClassifier(algorithm='brute', weights='distance')


# In[11]:


def NBModel():
    return naive_bayes.GaussianNB()

# In[12]:


def SVMModel_linear():
    return svm.SVC(kernel='linear')

def SVMModel_rbf():
    return svm.SVC(kernel='rbf', max_iter = 1000)

def SVMModel_LinearSVC():
    return svm.LinearSVC(max_iter = 5000)

# In[13]:


def NNModel(modelType):
    if modelType == 'simple':
        return neural_network.MLPClassifier(hidden_layer_sizes=(10, 10))
    elif modelType == 'complex':
        return neural_network.MLPClassifier(hidden_layer_sizes=(20, 20, 20, 20, 20, 20), alpha=0, solver='lbfgs', max_iter=1500)
    else:
        raise RuntimeError('Model must be of type linear or rbf')


# In[ ]:

model_list = [SVMModel_LinearSVC, SVMModel_rbf]

for m in model_list:
    runOnModelAndData(m, EMtrainDat, EMtestDat, EMtrainLabels, EMtestLabels)


# In[14]:


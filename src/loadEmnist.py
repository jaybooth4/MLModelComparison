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

def saveEmnistToNPY(data,filename):
    np.save(filename,data)

def loadEmnistFromNPY(filename):
    try:
        ret = np.load(filename)
    except FileNotFoundError:
        zipRef = zipfile.ZipFile('../data/EMNIST/balanced-data.zip')
        zipRef.extractall('../data/EMNIST')
        zipRef.close()
        ret = np.load(filename)

    return ret


def intToFloat(intData):
    floatData = intData.astype(float)
    return floatData


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


    
    
  



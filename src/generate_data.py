import numpy as np
from sklearn.model_selection import train_test_split

def generate_data():
  # Generate Data
  mean1 = [0,0]
  cov1 = [[1,0],[0,1]]
  dataSize1 = 2000
  gauss1 = np.random.multivariate_normal(mean1, cov1, dataSize1)
  labels1 = np.full(dataSize1, 1)

  mean2 = [2,2]
  cov2 = [[1,0],[0,1]]
  dataSize2 = 2000
  gauss2 = np.random.multivariate_normal(mean2, cov2, dataSize2)
  labels2 = np.full(dataSize1, 2)

  dataset = np.concatenate((gauss1, gauss2), axis=0)
  labels = np.concatenate((labels1, labels2), axis=0)

  data_train, data_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.20, random_state=42)
  return data_train, data_test, labels_train, labels_test
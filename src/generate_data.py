import numpy as np
from sklearn.model_selection import train_test_split

def generate_data(means, cov, datasize):
  # Generate Data
  numClasses = len(means)
  nDims = len(means[0])
  dataset = np.ndarray(0)
  labels = np.ndarray(0)
  gauss_ = np.ndarray([datasize*numClasses,nDims])
  labels_ = np.ndarray(datasize*numClasses)
  for i in range(numClasses):
    gauss_[i*datasize:(i+1)*datasize][:] = np.random.multivariate_normal(means[i], cov[i],datasize)
    labels_[i*datasize:(i+1)*datasize][:] = np.full(datasize,i)

  data_train, data_test, labels_train, labels_test = train_test_split(gauss_, labels_, test_size=0.20, random_state=42)
  return data_train, data_test, labels_train, labels_test
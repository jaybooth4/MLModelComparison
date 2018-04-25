'''
Code based on bag of words python implementation by Radim Řehůřek:
https://radimrehurek.com/data_science_python/
'''

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas
import csv
from textblob import TextBlob
import numpy as np

# Splits message into vector of lemmas
# Lemmas allow for words like "run", "ran",
# "runs", and "running" to be considered as
# the same feature
def split_into_lemmas(message):
    message = str(message)
    words = TextBlob(message).words
    return [word.lemma for word in words]

# Actual method that will convert entire spam
# message text file with (path) to a vector of
# TF-IDF frequencies that can be interpretted
# by our models
def bagOfWordsParser(path, testingSize = 0.1):
	print("Loading data from: ", path);

	# Converts the spam text file into an object of
	# labels and the messages associated to those labels
	messages = pandas.read_csv('../data/SPAM/SMSSpamCollection', sep='\t', 
							   quoting=csv.QUOTE_NONE,
	                           names=["label", "message"])

	numberOfSamples = len(messages)
	print("Number of samples: ", numberOfSamples)

	# Converts the messages into a vector of lemmas
	bagOfWordsTransformer = \
		CountVectorizer(analyzer = split_into_lemmas).fit(messages['message'])
	
	# Entire feature set is defined by the number of lemmas
	# in the vocabulary
	wordsInDictionary = len(bagOfWordsTransformer.vocabulary_)
	print("Number of words in vocabulary: ", wordsInDictionary)

	# Initiallize the labels and data arrays to correct shape
	labels = np.zeros((numberOfSamples))
	data = np.zeros((numberOfSamples, wordsInDictionary))

	# Shuffle dataset so that the messages can be randomly
	# sampled for testing/training pools
	messages = messages.sample(frac=1).reset_index(drop=True)

	for i in range(numberOfSamples):
		# Interpret a "spam" label as 1 and a "ham" label as 0.
		if messages.label[i] == "spam":
			labels[i] = 1
		elif messages.label[i] == "ham":
			labels[i] = 0
		else:
			print("Error - unsupported class!")
			exit(0)


		# Extract each message and transform that message into
		# a vector of TF-IDF frequencies for every lemma in the
		# defined vocabulary. The TF-IDF transformer allows us
		# to decrease the importance of words such as "is", "this",
		# or "as".
		message = messages.message[i]
		bagOfWords = bagOfWordsTransformer.transform([message])
		normalizedBagOfWords = TfidfTransformer().fit_transform(bagOfWords).toarray()
		data[i] = normalizedBagOfWords

	# Obtain the number of samples we want for testing
	numOfTestingData = int(numberOfSamples * testingSize)

	# Extract our texting/training data samples from
	# the full list
	trainData = labels[:-numOfTestingData]
	trainLabels = data[:-numOfTestingData]
	testData = labels[-numOfTestingData:]
	testLabels = data[-numOfTestingData:]

	# Return all samples
	return trainData, trainLabels, testData, testLabels


# For testing purposes: If this file is called in the command line,
# collect the data and print out the testing/training data/labels
if __name__ == '__main__':
	trainLabels, trainData, testLabels, testData = \
		 bagOfWordsParser('../data/SPAM/SMSSpamCollection')
	print(trainLabels.shape)
	print(trainData.shape)
	print(testLabels.shape)
	print(testData.shape)






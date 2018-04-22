'''
Code based on bag of words python implementation by Radim Řehůřek:
https://radimrehurek.com/data_science_python/
'''

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas
import csv
from textblob import TextBlob
import numpy as np

def split_into_lemmas(message):
    message = str(message)
    words = TextBlob(message).words
    return [word.lemma for word in words]


def bagOfWordsParser(path, testingSize = 0.1):
	print("Loading data from: ", path);
	messages = pandas.read_csv('../data/SPAM/SMSSpamCollection', sep='\t', 
							   quoting=csv.QUOTE_NONE,
	                           names=["label", "message"])

	numberOfSamples = len(messages)
	print("Number of samples: ", numberOfSamples)

	bagOfWordsTransformer = \
		CountVectorizer(analyzer = split_into_lemmas).fit(messages['message'])
	
	wordsInDictionary = len(bagOfWordsTransformer.vocabulary_)
	print("Number of words in vocabulary: ", wordsInDictionary)

	labels = np.zeros((numberOfSamples))
	data = np.zeros((numberOfSamples, wordsInDictionary))

	messages = messages.sample(frac=1).reset_index(drop=True)

	for i in range(numberOfSamples):
		if messages.label[i] == "spam":
			labels[i] = 1
		elif messages.label[i] == "ham":
			labels[i] = 0
		else:
			print("Error - unsupported class!")
			exit(0)

		message = messages.message[i]
		bagOfWords = bagOfWordsTransformer.transform([message])
		normalizedBagOfWords = TfidfTransformer().fit_transform(bagOfWords).toarray()
		data[i] = normalizedBagOfWords

	numOfTestingData = int(numberOfSamples * testingSize)

	trainData = labels[:-numOfTestingData]
	trainLabels = data[:-numOfTestingData]
	testData = labels[-numOfTestingData:]
	testLabels = data[-numOfTestingData:]

	return trainData, trainLabels, testData, testLabels

if __name__ == '__main__':
	trainLabels, trainData, testLabels, testData = \
		 bagOfWordsParser('../data/SPAM/SMSSpamCollection')
	print(trainLabels.shape)
	print(trainData.shape)
	print(testLabels.shape)
	print(testData.shape)






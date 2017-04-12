
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as pl
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import codecs
import pandas as pd 
from sklearn.feature_selection import SelectKBest, chi2

def ScikitExample_GaussianNB():
	Xtrain = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
	Ytrain = np.array([1, 1, 1, 2, 2, 2])
	x0,x1=zip(*Xtrain)
	pl.scatter(x0,x1,c=Ytrain,cmap=pl.cm.cool,s=20)
	
	clf = GaussianNB()
	clf.fit(Xtrain, Ytrain)

	X=[[-0.8, -1],[2.5,1.5],[0,0.5]]
	Y= clf.predict(X)
	x0,x1=zip(*X)
	pl.scatter(x0,x1,c=Y, cmap=pl.cm.cool,marker='x',lw=2)
	pl.show()

def ScikitExample_MultinomialNB():
	import numpy as np
	X = [[np.random.randint(5) for i in range(5)]+[np.random.randint(30) for i in range(5)] for j in range(10)]	\
	+[[np.random.randint(30) for i in range(5)]+[np.random.randint(5) for i in range(5)] for j in range(10)]
	y = np.array([1 for i in range(10)]+[2 for i in range(10)])
	for x in X:
		print x
	clf = MultinomialNB()
	clf.fit(X, y)
	ypred=clf.predict([[np.random.randint(5) for i in range(5)]+[np.random.randint(30) for i in range(5)] ,
		[np.random.randint(30) for i in range(5)]+[np.random.randint(5) for i in range(5)]])

def Text_MultinomialNB():
	df = pd.read_csv("tweets.csv")
	#print df['handle']
	vectorizer = TfidfVectorizer(strip_accents='unicode',stop_words='english')
	X_train = vectorizer.fit_transform(df['text'])
	y_train=df['handle']
	feature_names = vectorizer.get_feature_names()
	ch2 = SelectKBest(chi2, k=1000)
	X_train = ch2.fit_transform(X_train,y_train )
	feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
	X_test = vectorizer.transform(df['text'])
	X_test = ch2.transform(X_test)
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print Counter(zip(y_train,y_pred))

Text_MultinomialNB()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("emails.csv")

print(df.info())

print(df.describe())

ham = df[df['spam']==0]
spam = df[df['spam']==1]

sns.countplot(df['spam'],label='Count plot of Ham vs Spam')

print("The ham precentage is ",len(ham)/len(df),"%")
print("The spam percentage is ",len(spam)/len(df),"%")

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
spamham = cv.fit_transform(df['text'])

print(cv.get_feature_names())

label = df['spam'].values

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(spamham,label)
test = ['Hi this is anil. Welcome to India pal', 'Do you want some free cookies? ']
testcv=cv.transform(test)
testpred = nb.predict(testcv)
testpred

x=spamham
y=label

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
nb_c = MultinomialNB()
nb_c.fit(xtrain,ytrain)

ypredtrain = nb_c.predict(xtrain)

from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(ytrain,ypredtrain)

sns.heatmap(cm,annot=True)

ypredtest = nb_c.predict(xtest)

cm = confusion_matrix(ypredtest,ytest)

sns.heatmap(cm,annot=True)

test = ['Back in amsterdam buddy. Looking forward to meet you !','Earn 20$ online just using these simple steps. To learn more click on the link below.']
testcv = cv.transform(test)
ypredcvtest = nb_c.predict(testcv)

print(ypredcvtest)
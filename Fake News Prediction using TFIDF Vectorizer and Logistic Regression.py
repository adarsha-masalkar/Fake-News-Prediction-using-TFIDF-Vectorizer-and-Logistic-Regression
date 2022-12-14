import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# print(stopwords.words('English'))

df = pd.read_csv('train.csv')
df.isnull().sum()
df = df.fillna('')

df['content'] = df['author'] + ' ' + df['title']

# Stemming is the process of reducing a word to its root word.
# Eg. singer, singing -> sing

stemmer = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = [stemmer.stem(word) for word in stemmed_content.lower().split() if not word in stopwords.words('english')]
    return ' '.join(stemmed_content)


df['content'] = df['content'].apply(stemming)

X = df['content'].values
Y = df['label'].values

vectorizer = TfidfVectorizer()

vectorizer.fit(X)

X = vectorizer.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2, stratify = Y)

model = LogisticRegression()

model.fit(x_train, y_train)

predict = model.predict(x_train)

accuracy_score = accuracy_score(predict, y_train)

print('Accuracy score is ', accuracy_score)


def predictor(n):
    x_new = x_test[n]
    prediction = model.predict(x_new)

    if prediction[0] == 0:
        print('Real News')
    else:
        print('Fake News')

        
predictor(0)
predictor(1)
predictor(2)
predictor(3)
predictor(4)

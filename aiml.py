import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

nltk.download('punkt')
nltk.download('stopwords')

news_dataset = pd.read_csv('content/train.csv')
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)
X = news_dataset['content'].values
Y = news_dataset['label'].values

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
X_old = vectorizer.fit_transform(X)
X = transformer.fit_transform(X_old)

warnings.filterwarnings('ignore')
parameters = {
    'penalty' : ['l1','l2'],
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

logreg = LogisticRegression()
model = GridSearchCV(logreg,
                   param_grid = parameters,
                   scoring='accuracy',
                   cv=10)

model.fit(X_train, Y_train)
print("Tuned Hyperparameters :", model.best_params_)
print("Accuracy :",model.best_score_)

model = LogisticRegression(C = 100.0, penalty = 'l1', solver = 'liblinear')
model.fit(X_train,Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
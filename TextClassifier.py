# useful link : http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py

import Preprocessor as preproc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics, preprocessing, cross_validation
import Evaluator

def mergeTextColumns(data, columnNames):
    return data[columnNames].apply(lambda x: ' '.join(x), axis=1)

data = preproc.data
headers = preproc.getHeaders()

targetColumn = 'gross'
min_max_scaler = preprocessing.MinMaxScaler()

textColumnNames = ['plot_keywords', 'genres', 'director_name', 'actor_1_name', 'movie_title', 'actor_3_name', 'language', 'country']
X = mergeTextColumns(data, textColumnNames)
y = min_max_scaler.fit_transform(data[targetColumn])
data_train, data_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(data_train)
X_test = vectorizer.transform(data_test)


clf = RandomForestRegressor()
        
Evaluator.printEvaluation(clf, X_train, y_train, X_test, y_test, "regressor")
Evaluator.printCrossValidationResult(clf, TfidfVectorizer().fit_transform(X), y, "regressor")
Evaluator.evaluateAllRegressionByCrossValidation(X, y)

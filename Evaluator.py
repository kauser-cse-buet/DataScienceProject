import Preprocessor as preproc
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics, preprocessing, cross_validation

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import (linear_model,cross_validation)
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
import time

RegressorModel = ['RandomForestRegressor']
ClassifierModel = []

def printEvaluation(clf, X_train, y_train, X_test, y_test, CLFType):    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if CLFType.lower() == 'regressor':
        print("MSE: ", metrics.mean_squared_error(y_true=y_test, y_pred=y_pred))
        print("MAE:", metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred))
    elif CLFType.lower() == 'classifier_binary':
        pass
    elif CLFType.lower() == 'classifier_multiclass':
        pass
        
def printCrossValidationResult(clf, X, y, CLFType):    
    print("Cross Validation")
    if CLFType.lower() == 'regressor':
        result = cross_validation.cross_val_score(clf, X, y, scoring="neg_mean_squared_error", cv = 5)
        print("MSE: ", abs(result.mean()))
        result = cross_validation.cross_val_score(clf, X, y, scoring="neg_mean_absolute_error", cv = 5)
        print("MAE: ", abs(result.mean()))
    elif CLFType.lower() == 'classifier_binary':
        pass
    elif CLFType.lower() == 'classifier_multiclass':
        pass

def evaluateAllRegressionByCrossValidation(X, y):
    for model in RegressorModel:
        if model.lower() == 'randomforestregressor':
            clf = RandomForestRegressor()

        printCrossValidationResult(clf, X, y, 'regressor')

    # For classification
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    start = time.time()
    clf.fit(X_train, y_train)
    # print("Accuracy on training set:", clf.score(X_train, y_train))
    # print("Accuracy on testing set:", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", metrics.classification_report(y_test, y_pred))
    # print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
    # print("Running time:", time.time()-start)

def evaluateClassification(x,y,train_size=0.5):
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.5)
    print('------------Logistic regrassion  l2')
    train_and_evaluate(LogisticRegression(penalty='l2', C=1),x_train,x_test,y_train,y_test)
    print('------------Logistic regrassion  l1')
    train_and_evaluate(LogisticRegression(penalty='l1', C=1),x_train,x_test,y_train,y_test)
    print('------------Decision Tree method')
    train_and_evaluate(DecisionTreeClassifier(),x_train,x_test,y_train,y_test)
    print('------------Extra Trees method')
    train_and_evaluate(ExtraTreesClassifier(),x_train,x_test,y_train,y_test)
    print('------------Random Forest method')
    train_and_evaluate(RandomForestClassifier(),x_train,x_test,y_train,y_test)
    print('------------GaussianNaiveBayes method')
    train_and_evaluate(GaussianNB(),x_train,x_test,y_train,y_test)

def crossvalidationClassification(x, y):
    print('------------Decision Tree method')
    evaluateCrossVal(DecisionTreeClassifier(), x, y)
    print('------------Extra Trees method')
    evaluateCrossVal(ExtraTreesClassifier(), x, y)
    print('------------Random Forest method')
    evaluateCrossVal(RandomForestClassifier(), x, y)
    print('------------Logistic regrassion  l2')
    evaluateCrossVal(LogisticRegression(penalty='l2', C=1),x,y)
    print('------------Logistic regrassion  l1')
    evaluateCrossVal(LogisticRegression(penalty='l1', C=1),x,y)
    print('------------GaussianNaiveBayes method')
    evaluateCrossVal(GaussianNB(), x, y)

def evaluateCrossVal(classifier, x, y):    
    classifier.fit(x, y)
    cv_f1_scores = cross_validation.cross_val_score(classifier, x, y, cv=5, scoring='f1')
    cv_precision_scores = cross_validation.cross_val_score(classifier, x, y, cv=5,scoring='precision')
    cv_recall_scores = cross_validation.cross_val_score(classifier, x, y, cv=5,scoring='recall')
    print('vectorization Precision=',cv_precision_scores.mean())
    print('vectorization Recall=',cv_recall_scores.mean())
    print('vectorization F1=', cv_f1_scores.mean())

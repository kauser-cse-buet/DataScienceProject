import Preprocessor as preproc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics, preprocessing, cross_validation


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



    

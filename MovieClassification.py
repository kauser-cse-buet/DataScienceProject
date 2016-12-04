import pandas
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
import time
import Preprocessor

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
	start = time.time()
	clf.fit(X_train, y_train)
	print("Accuracy on training set:", clf.score(X_train, y_train))
	print("Accuracy on testing set:", clf.score(X_test, y_test))
	y_pred = clf.predict(X_test)
	print("Classification Report:\n", metrics.classification_report(y_test, y_pred))
	print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
	print("Running time:", time.time()-start)

def evaluate(x,y,train_size=0.5):
	x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.5)
	print('Decision Tree method')
	train_and_evaluate(DecisionTreeClassifier(),x_train,x_test,y_train,y_test)
	print('Extra Trees method')
	train_and_evaluate(ExtraTreesClassifier(),x_train,x_test,y_train,y_test)
	print('Random Forest method')
	train_and_evaluate(RandomForestClassifier(),x_train,x_test,y_train,y_test)
	print('SVC method')
	train_and_evaluate(svm.SVC(),x_train,x_test,y_train,y_test)
	print('Logit method')
	train_and_evaluate(LogisticRegressionCV(),x_train,x_test,y_train,y_test)
	print('GaussianNaiveBayes method')
	train_and_evaluate(GaussianNB(),x_train,x_test,y_train,y_test)

def numerical_to_binary(y, cut_off):
	y.iloc[[i for i,v in enumerate(y) if v < cut_off]]=0
	y.iloc[[i for i,v in enumerate(y) if v >= cut_off]]=1
	return y


data = pandas.read_csv('movie_metadata.csv')
features_allnum = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_1_facebook_likes', 'gross', 'num_user_for_reviews', 'imdb_score', 'budget', 'title_year', 'aspect_ratio', 'movie_facebook_likes']
features_x = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_1_facebook_likes', 'gross', 'num_user_for_reviews', 'budget', 'title_year', 'aspect_ratio', 'movie_facebook_likes']
feature_y = 'imdb_score'

data = data[features_allnum]

data = data.dropna(how='any')

#x = data[features_x]
#y =data[feature_y]
x, y = Preprocessor.getX_y(feature_y, features_allnum)
print(y[1:5])

cut_off = 8
y=numerical_to_binary(y, cut_off)

print(y[1:5])

print(len(x))
print(len(y))

evaluate(x, y, 0.5)
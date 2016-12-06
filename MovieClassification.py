# Author: Sayma Akther
# Classification of High rated movie, if IMDB rating greater than 8

import pandas
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LassoLarsIC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, preprocessing, cross_validation
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

import time
import Preprocessor
import Evaluator


def numerical_to_binary(y, cut_off):
	y.iloc[[i for i,v in enumerate(y) if v < cut_off]]=0
	y.iloc[[i for i,v in enumerate(y) if v >= cut_off]]=1
	return y

def token(text):
    return(text.split("|"))

def scaler(data,target):
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    target = pd.DataFrame(scaler.fit_transform(target))
    return data,target


data = pandas.read_csv('movie_metadata.csv')
features_allnum = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_1_facebook_likes', 'gross', 'num_user_for_reviews', 'imdb_score', 'budget', 'title_year', 'aspect_ratio', 'movie_facebook_likes']
features_x = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_1_facebook_likes', 'gross', 'num_user_for_reviews', 'budget', 'title_year', 'aspect_ratio', 'movie_facebook_likes']
feature_y = 'imdb_score'
list_fig = ['director_facebook_likes','gross','num_voted_users','num_critic_for_reviews','num_user_for_reviews','budget',\
            'movie_facebook_likes','duration','actor_3_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes',\
           'cast_total_facebook_likes','facenumber_in_poster','profit']
data = Preprocessor.data
data['profit'] = data['gross'] - data['budget']

x, y = Preprocessor.getX_y(feature_y, features_allnum)
# x,y = scaler(x,y)
print("Classification of High rated movie, if IMDB rating greater than 8")

cut_off = 8
y=numerical_to_binary(y, cut_off)
Evaluator.evaluateClassification(x, y, 0.5)
# Evaluator.crossvalidationClassification(x, y)

# Adding text features
vectorizer = TfidfVectorizer(tokenizer=token)
genre = vectorizer.fit_transform(data['genres'])
genres_list = ["genres_"+ i for i in vectorizer.get_feature_names()]

vectorizer_pk = TfidfVectorizer(max_features=50,tokenizer=token)
pk = vectorizer_pk.fit_transform(data['plot_keywords'])
pk_list = ["pk_"+ i for i in vectorizer_pk.get_feature_names()]

vectorizer_c = TfidfVectorizer()
c = vectorizer_c.fit_transform(data['country'])
c_list = ["c_"+ i for i in vectorizer_c.get_feature_names()]

x_all = np.hstack([data.ix[:,list_fig],genre.todense(),pk.todense(),c.todense()])

print('-----------------------Numerical + categorical features----------------------------------------')
Evaluator.evaluateClassification(x_all, y, 0.5)


# Author: Sayma Akther
# Classification of profitable movie, if gross is greater than budget

import pandas
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LassoLarsIC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, preprocessing, cross_validation
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

import time
import Preprocessor
import Evaluator


def numerical_to_binary(y, cut_off):    
	y.iloc[[i for i,v in enumerate(y) if v >= cut_off]]=1
	y.iloc[[i for i,v in enumerate(y) if v < cut_off]]=0
	return y

def token(text):
    return(text.split("|"))

def scaler(data,target):
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    target = pd.DataFrame(scaler.fit_transform(target))
    return data,target


data = pandas.read_csv('movie_metadata.csv')
features_allnum = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_1_facebook_likes', 'gross', 'num_user_for_reviews', 'imdb_score', 'budget', 'title_year', 'aspect_ratio', 'movie_facebook_likes']
features_x = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_1_facebook_likes', 'gross', 'num_user_for_reviews', 'budget', 'title_year', 'aspect_ratio', 'movie_facebook_likes']
feature_y = 'gross'
list_fig = ['director_facebook_likes','num_voted_users','num_critic_for_reviews','num_user_for_reviews','budget',\
            'movie_facebook_likes','duration','actor_3_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes',\
           'cast_total_facebook_likes','facenumber_in_poster']
data = Preprocessor.data
data['profit'] = data['gross'] - data['budget']

x, y = Preprocessor.getX_y(feature_y, features_allnum)
# x,y = scaler(x,y)
y = data['profit']
print("Classification of profitable movie, if gross is greater than budget")

cut_off = 0
y=numerical_to_binary(y, cut_off)
Evaluator.evaluateClassification(x, y, 0.5)
# Evaluator.crossvalidationClassification(x, y)

# Adding text features
vectorizer = TfidfVectorizer(tokenizer=token)
genre = vectorizer.fit_transform(data['genres'])
genres_list = ["genres_"+ i for i in vectorizer.get_feature_names()]

vectorizer_pk = TfidfVectorizer(max_features=50,tokenizer=token)
pk = vectorizer_pk.fit_transform(data['plot_keywords'])
pk_list = ["pk_"+ i for i in vectorizer_pk.get_feature_names()]

vectorizer_c = TfidfVectorizer()
c = vectorizer_c.fit_transform(data['country'])
c_list = ["c_"+ i for i in vectorizer_c.get_feature_names()]

x_all = np.hstack([data.ix[:,list_fig],genre.todense(),pk.todense(),c.todense()])

print('-----------------------Numerical + categorical features----------------------------------------')
Evaluator.evaluateClassification(x_all, y, 0.5)

# useful link : http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py

import Preprocessor as preproc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics, preprocessing, cross_validation
import Evaluator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

%pylab inline

data = preproc.data
headers = preproc.getHeaders()


def getDataGroupByAgg(columnName, target, aggMethodName, topCount=10):
    dataGroupBy = data.groupby(columnName)
    dataGroupByAgg = None
    if aggMethodName.lower() == 'mean':
        dataGroupByAgg = dataGroupBy.mean()
    if aggMethodName.lower() == 'median':
        dataGroupByAgg = dataGroupBy.median()
    if aggMethodName.lower() == 'sum':
        dataGroupByAgg = dataGroupBy.sum()

    return dataGroupByAgg[[target]].sort(target, ascending=False).head(10)


nameList = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']
targetList = ['gross', 'imdb_score', 'num_voted_users', 'num_critic_for_reviews', 'num_user_for_reviews', 'movie_facebook_likes']
aggFuncList = ['mean', 'median', 'sum']


for i in range(len(nameList)):
    for j in range(len(targetList)):
        for k in range(len(aggFuncList)):
            dataGroupByAgg = getDataGroupByAgg(nameList[i], targetList[j], aggFuncList[k], 10)
            dataGroupByAgg.plot(kind='barh', title= nameList[i] + " vs "+ targetList[j] + "(" + aggFuncList[k] + ")")
            plt.show()

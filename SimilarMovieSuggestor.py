# Got idea and help from https://www.kaggle.com/airalex/d/deepmatrix/imdb-5000-movie-dataset/recommend-movie-with-clustering

import pandas as pd
import numpy as np
import Preprocessor
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

data = Preprocessor.data
featuresSelected = ['plot_keywords', 'movie_title', 'actor_1_name',
                       'actor_2_name', 'actor_3_name', 'director_name', 'imdb_score', 'genres', 'gross']
data_x = data.ix[:, featuresSelected]

dataProcessed = data_x.drop_duplicates(['movie_title'])
dataProcessed = dataProcessed.reset_index(drop=True)

castNCrewList = []
castNCrewTypeList = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']
for i in range(dataProcessed.shape[0]):
    nameList = []
    for castNCrew in castNCrewTypeList:
        name = dataProcessed.ix[i, castNCrew].replace(" ", "_")
        nameList.append(name)
    castNCrewList.append("|".join(nameList))
dataProcessed['castNCrew'] = castNCrewList

def token(text):
    return (text.split("|"))

def getFeatureNameList(termDocuments, featureType):
    return [featureType + "_" + i for i in termDocuments.get_feature_names()]

def getFeaturesAndList(featureName):
    vectorizer = CountVectorizer(max_features=100, tokenizer=token)
    features = vectorizer.fit_transform(dataProcessed[featureName])
    featuresList = getFeatureNameList(vectorizer, featureName[:3])

    return features, featuresList

kwrds, kwrdsList = getFeaturesAndList("plot_keywords")
genre, genreList = getFeaturesAndList("genres")
castNCrew, castNCrewList = getFeaturesAndList("castNCrew")

dataCluster = np.hstack([kwrds.todense(), genre.todense(), castNCrew.todense() * 2])
criterionList = kwrdsList + genreList + castNCrewList

model = KMeans(n_clusters=100)
label = model.fit_predict(dataCluster)
labelDF = pd.DataFrame({"category": label}, index=dataProcessed['movie_title'])

def suggester(movieTitle, noOfSuggestion=3, featureToSortOn = 'imdb_score'):
    if movieTitle in list(dataProcessed['movie_title']):
        movieCluster = labelDF.ix[movieTitle, 'category']
        similarMovies = dataProcessed.ix[list(labelDF['category'] == movieCluster), [featureToSortOn, 'movie_title']]
        similarMoviesSorted = similarMovies.sort_values([featureToSortOn], ascending=[0])
        similarMoviesSorted = similarMoviesSorted[similarMoviesSorted['movie_title'] != movieTitle]
        if noOfSuggestion > similarMoviesSorted.shape[0]:
            noOfSuggestion = similarMoviesSorted.shape[0]
        suggestedMovieList = list(similarMoviesSorted.iloc[range(noOfSuggestion), 1])
        return suggestedMovieList
    else:
        return None

def getMovieSimilarTitle(title):
    maxLen = 0
    movieSimilarTitle = []
    for movie in set(dataProcessed['movie_title']):
        for i in range(len(title)):
            for j in range(i, len(title)):
                if title[i: j + 1].lower() in movie.lower() and len(title[i: j + 1]) > maxLen:
                    maxLen = len(title[i: j + 1])
                    movieSimilarTitle.append(movie)

    movieSimilarTitle.reverse()

    if len(movieSimilarTitle) > 3:
        return movieSimilarTitle[:3]
    else:
        return movieSimilarTitle


# # Got idea and help from https://blog.dominodatalab.com/interactive-dashboards-in-jupyter/

from ipywidgets import widgets, interact, interactive, fixed
from IPython.display import display

movieNameText = widgets.Text(description="Movie Name: ", width=200)
SimilarMovieNameText = widgets.Textarea(description='Similar Movies:')
featureToSortOnText = widgets.Text(description="gross/IMDB: ", width=200)
display(movieNameText)
display(featureToSortOnText)
display(SimilarMovieNameText)



def handle_submit(sender):
    if featureToSortOnText.value == 'gross':
        featureToSortOn = 'gross'
    else:
        featureToSortOn = 'imdb_score'

    recommend_movie = suggester(movieNameText.value, 10, featureToSortOn)
    if recommend_movie is None:
        movieSimilarTitle = getMovieSimilarTitle(movieNameText.value)
        SimilarMovieNameText.value = "Incorrect Movie Name!!!!!" + " You can try: " + str(movieSimilarTitle)
    else:
        SimilarMovieNameText.value = str(recommend_movie)


movieNameText.on_submit(handle_submit)
featureToSortOnText.on_submit(handle_submit)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

imdb_movie = pd.read_csv("movie_metadata.csv")

def show_corr_numerical_values():
    features = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_1_facebook_likes', 'gross', 'num_user_for_reviews', 'budget', 'title_year', 'imdb_score', 'aspect_ratio', 'movie_facebook_likes']
    imdb_movie_sf = imdb_movie[features]

    imdb_movie_sf = imdb_movie_sf.dropna(how='any')

    # imdb_movie = imdb_movie.dropna(how='any')
    # print(imdb_movie.head(5))
    print(pd.DataFrame({'key': np.arange(0, len(features)), 'value': features}))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    corr = imdb_movie_sf.corr()
    c = ax.matshow(corr)

    # ax.set_xticklabels(list(imdb_movie.dtypes.keys()))
    # ax.set_yticklabels(list(imdb_movie.dtypes.keys()))

    fig.colorbar(c)
    plt.show()

# print(list(imdb_movie.dtypes.keys()))

# show_corr_numerical_values()

print(imdb_movie)




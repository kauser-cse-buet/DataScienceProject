#Anjana Tiha
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


imdb_movie = pd.read_csv("movie_metadata.csv")

imdb_movie['actor_1_name'] = imdb_movie['actor_1_name'].fillna('None').astype('category')
imdb_movie['actor_1_facebook_likes'] = imdb_movie['actor_1_facebook_likes'].fillna(0.0).astype(np.float)
#imdb_movie['actor_2_name']=imdb_movie['actor_2_name'].fillna(0).astype('category')
imdb_movie['actor_2_facebook_likes'] = imdb_movie['actor_2_facebook_likes'].fillna(0.0).astype(np.float)
#imdb_movie['actor_3_name']=imdb_movie['actor_3_name'].fillna(0).astype('category')
imdb_movie['actor_3_facebook_likes'] = imdb_movie['actor_3_facebook_likes'].fillna(0.0).astype(np.float)
#imdb_movie['director_name']=imdb_movie['director_name'].fillna(0).astype('category')
imdb_movie['director_facebook_likes'] = imdb_movie['director_facebook_likes'].fillna(0.0).astype(np.float)
imdb_movie['cast_total_facebook_likes'] = imdb_movie['cast_total_facebook_likes'].fillna(0.0).astype(np.float)
imdb_movie['budget'] = imdb_movie['budget'].fillna(0.0).astype(np.float)
imdb_movie['gross'] = imdb_movie['gross'].fillna(0.0).astype(np.float)

target = imdb_movie['gross']
raw_training_data = imdb_movie.drop('gross', axis = 1)


#actor_set_1 = set(imdb_movie['actor_1_name'])
#actor_list_1 = list(actor_set_1)
actor_list_1 = list(imdb_movie['actor_1_name'])

lbl_enc = LabelEncoder()
label_actor_1 = lbl_enc.fit_transform(actor_list_1)
ht_enc = OneHotEncoder()
catagory_actor_1 = ht_enc.fit_transform(label_actor_1)
#catagory_actor_1.reshape(-1, 1)

data_list = list(zip( raw_training_data['director_facebook_likes'], raw_training_data['actor_1_facebook_likes'],
    raw_training_data['actor_2_facebook_likes'], raw_training_data['actor_3_facebook_likes'], 
    raw_training_data['cast_total_facebook_likes'], raw_training_data['budget'], catagory_actor_1))

data = np.array(data_list)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.3, random_state = 0)

regr = SVR()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

mean_abs_error = mean_absolute_error(y_test, y_pred)
print("mean_absolute_error", mean_abs_error)
print("original:\n", y_test[:5], "prediction:", y_pred[:5])

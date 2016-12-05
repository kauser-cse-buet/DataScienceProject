#Movie Gross predictor:

We are building an application in python which will predict gross of a movie based on data from movie database. 


Right now, We are in skeleton steps. 


we need to discuss the following:
- What is our goal. What are we going to model
- What are the features that can be used
- Develop various visualizations to help indicate feature selection
- Finally what machine learning model can we use
- How are we going to present the results in the documentation


The project will have the following modules:
- Preprocessor
- Model
- Visualization
- Validation

## Preprocessor 

- Drop row which has "NA"
- Drop column which maximum values are "0". Set maximum values percentage like >= 50 percentage.

## Model
## Visualization
1. Top and bottom popular movie (by imdb score) grouped by country.
2. Find the 10 most liked directors, actors.
3. Most profitable movies grouped by countries. 
4. Duration and movies grouped by year/country.
5.  
## Validation


Anjana Tiha's part:
Movie gross prediction:

Preprocessing numerical, categorical data and text data especially for gross prediction
We have computed gross prediction. Both numerical and categorical data has been used for gross prediction/regression.
Top actor1, actor2, actor3, director, country, content rating, language by mean gross have been visualized.
Different models have been applied to test performance.
Among them, Random forest regressor and decision tree regressor worked well.
Using SVM did not work due to high sparcisty of available data.
Features used: actor1 facebook likes, actor2 facebook likes, actor3 facebook likes, director facebook likes,
Categorical features - actor1 name, actor2 name, actor3 name, director name, country, content rating, language
models have been validated usung different regressor evaluation functions
for evaluation Mean Absolute Error, Mean Squared Error, Median Absolute Error, Explained Var Score, R^2 score have been calculated
regression score was followings:
Mean Absolute Error    : 23948803.9178
Mean Squared Error     : 2.11567515688e+15
Median Absolute Error  : 10256857.0
Explained Var Score    : 0.477276813998
R^2 Score              : 0.459857366497


## text feature inclusions
- if we add text plot_keywords, genres, as features. 
- it improves result score and reduce error. 
- visualisation: 
- corelation between plot_keywords, genres and imdb.
- related imdb score has similar genres. 
## statistics 
- x wise movie budget 
- x wise movie profit
- x wise imdb score
- x = country , year, genre

- genre wise budget, profit, imdb score.







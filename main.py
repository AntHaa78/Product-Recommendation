
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # disable a false positive warning line 116
import random
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
#from sklearn.externals import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

""" from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings; warnings.simplefilter('ignore')
%matplotlib inline """

#load data and give colum names
columns = ['userID', 'productID', 'ratings', 'timestamp']
videogames_df=pd.read_csv('Video_Games.csv', names=columns)
#print(videogames_df.info())

#drop timestamp
videogames_df.drop(columns=['timestamp'], inplace=True)
rows, columns = videogames_df.shape
#print(f"Number of rows: {rows}  | Number of columns: {columns}")

# Taking subset (large dataset)
videogames_df1=videogames_df.iloc[:10000]
del videogames_df
#print(videogames_df1.info())

#summary statistics of rating variable
#print(videogames_df1['ratings'].describe())
#print(f"Min rating is {videogames_df1.ratings.min()} and max rating is {videogames_df1.ratings.max()}")

#checking for missing values
#print("Number of missing values across columns:\n", videogames_df1.isnull().sum())
# -> no missing values


# rating distribution graphs
""" print(videogames_df1['ratings'].value_counts())
videogames_df1.groupby('userID')['ratings'].count().sort_values(ascending=False)
ax = sns.countplot(x='ratings', data=videogames_df1, palette = 'Set1', edgecolor = 'black')
for i in range(0,5):
    ax.bar_label(ax.containers[i])
ax.set_ylabel("Total count of ratings")
plt.show() """

# Number of unique user id  in the data
print('Number of unique users in videogames data = ', videogames_df1['userID'].nunique())
# Number of unique product id  in the data
print('Number of unique products in videogames data = ', videogames_df1['productID'].nunique())

#Check the top 10 users based on ratings
most_rated=videogames_df1.groupby('userID').size().sort_values(ascending=False)[:10]
print('Top 10 users based on number of ratings: \n',most_rated)

# Take users that provided > 200 ratings
counts = videogames_df1.userID.value_counts()
videogames_df1_final = videogames_df1[videogames_df1.userID.isin(counts[counts>200].index)]
print('Number of users that provided more than 200 ratings: ', videogames_df1_final['userID'].nunique())
print('Number of unique products in final data: ', videogames_df1_final['productID'].nunique())
print('Number of products in final data: ', len(videogames_df1_final))

# checking top 10 products with most ratings
#print(videogames_df1_final.groupby("productID").size().sort_values(ascending=False)[:10])

#construction of pivot table. We have some duplicates users/productID so using the mean of ratings(aggfunc), if a product has no rating from a user -> 0 (fillna0)
#videogames_df1_final.set_index(['userID', 'productID', 'ratings'], append=True)
final_ratings_matrix = videogames_df1_final.pivot_table(index='userID', columns ='productID', values='ratings', aggfunc='mean').fillna(0)
#print(final_ratings_matrix.head())
print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)

#Density of the rating matrix
num_of_ratings_given = np.count_nonzero(final_ratings_matrix)
print("Number of ratings given: ", num_of_ratings_given)
num_of_ratings_possible = final_ratings_matrix.shape[0]*final_ratings_matrix.shape[1]
print("Possible total of ratings: ", num_of_ratings_possible)
density = (num_of_ratings_given/num_of_ratings_possible)*100
print(f"The density of the matrix is: {density:.2f}%")
# -> Sparse matrix, unfortunately not many products have ratings from different users

#Split the data randomly into train and test datasets, ratio 70:30
train_data, test_data = train_test_split(videogames_df1_final, test_size = 0.3, random_state=0)
#print(train_data.head())
print('Shape of training data: ',train_data.shape)
print('Shape of testing data: ',test_data.shape)

# ----------------------------------------------------------------------
# Popularity recommender model

# Count number of ratings and average rating for each unique product as a recommendation score
train_data_grouped = train_data.groupby('productID').agg({'userID': 'count','ratings': 'mean'}).reset_index()
train_data_grouped.rename(columns={'ratings': 'average rating'}, inplace=True)
train_data_grouped.rename(columns = {'userID': 'score'}, inplace=True)
# Sort the products on based on a score, first number of ratings and then average rating, check top 10
train_data_sort = train_data_grouped.sort_values(['score', 'average rating'], ascending=False)
print(train_data_sort.head(10))
# We see that first 5 products have 4+ reviews and average score >3.75, this will be our 5 recommended products
train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first') # add of rank column
popularity_recommendations = train_data_sort.head(5)
print(popularity_recommendations)
print('\n')


#Use of popularity based recommender model to make predictions
def recommend(user_id):
    user_recommendations = popularity_recommendations
    # add user_id column for which the recommendation is generated
    user_recommendations['userId'] = user_id
    #bring user_id column to the front
    cols = user_recommendations.columns.tolist() 
    cols = cols[-1:] + cols[:-1] 
    user_recommendations = user_recommendations[cols] 
          
    return user_recommendations      

random_users = random.sample(range(1, 500), 3)
#for i in random_users:  
#    print("The list of recommendations for the userId: %d\n" %(i))
#    print(recommend(i)) 
#    print("\n") 

# Conclusion: We see that every user gets the same recommendations (normal) -> non-personalized recommender model.
    
#----------------------------------------------------------------------
# Collaborative Filtering recommender model

videogames_df_CF = pd.concat([train_data, test_data]).reset_index()
#print('\n', videogames_df_CF.head())

#Matrix with row = user and column = product
pivot_df = videogames_df_CF.pivot_table(index='userID', columns='productID', values = 'ratings', aggfunc='mean').fillna(0)
#print(pivot_df.head())
print('Shape of the pivot table: ', pivot_df.shape)

# Define user index from 0 to 10
pivot_df['user_index'] = np.arange(pivot_df.shape[0])
pivot_df.set_index(['user_index'], inplace=True)
print(pivot_df.head())
# Another sparse matrix. Lets use matrix factorization to build our model

# Matrix Factorization using Singular Value Decomposition (SVD). (pivot df dataframe->np.ndarray)

U, s, V = svds(pivot_df.to_numpy(), k=10)
#print("U matrix(users): \n", U, "\n Sigma matrix: \n", s, "\nV matrix (products): \n", V)

# Sigma is not diagonal
s = np.diag(s)
#print('Diagonal matrix: \n',s)

#Predicted ratings
all_user_predicted_ratings = np.dot(np.dot(U, s), V) 
# Convert predicted ratings to dataframe
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = pivot_df.columns)
#print(preds_df.head())

# Recommend the videogames with the highest predicted ratings
def recommend_videogames(user, pivot_df, preds_df, num_recommendations):
    sorted_user_ratings = pivot_df.loc[user-1].sort_values(ascending=False)
    sorted_user_predictions = preds_df.loc[user-1].sort_values(ascending=False)
    result = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
    result.index.name = 'Recommended items'
    result.columns = ['user_ratings', 'user_predictions']
    result = result.loc[result.user_ratings==0]
    result = result.sort_values('user_predictions', ascending=False)
    print('\nHere are the recommended videogames for user number: ', user)
    print(result.head(num_recommendations))


num_recommendations = 5
users = [1,5,8]
#[recommend_videogames(i, pivot_df, preds_df, num_recommendations) for i in users]
# -> Different recommended items depending on user and their past behaviour

# Evaluation of model, Root Mean Squared Error (RMSE)

final_ratings_matrix.mean() # average  actual rating for each item
preds_df.mean() #average predicted rating for each item

rmse_df = pd.concat([final_ratings_matrix.mean(), preds_df.mean()], axis=1)
rmse_df.columns = ['avg_actual_ratings', 'avg_predicted_ratings']
rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)
#print(rmse_df.head())

RMSE = round((((rmse_df.avg_actual_ratings - rmse_df.avg_predicted_ratings) ** 2).mean() ** 0.5), 3)
print('\nRMSE SVD model = ', RMSE)

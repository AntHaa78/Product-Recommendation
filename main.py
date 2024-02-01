
import numpy as np
import pandas as pd
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
videogames_df1=videogames_df.iloc[:3000]
del videogames_df
#print(videogames_df1.info())

#summary statistics of rating variable
#print(videogames_df1['ratings'].describe())
#print(f"Min rating is {videogames_df1.ratings.min()} and max rating is {videogames_df1.ratings.max()}")

#checking for missing values
#print("Number of missing values across columns:\n", videogames_df1.isnull().sum())

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

# Take users that provided > 100 ratings
counts = videogames_df1.userID.value_counts()
videogames_df1_final = videogames_df1[videogames_df1.userID.isin(counts[counts>100].index)]
print('Number of users that provided more than 100 ratings: ', videogames_df1_final['userID'].nunique())
print('Number of unique products in final data: ', videogames_df1_final['productID'].nunique())
print('Number of products in final data: ', len(videogames_df1_final))

#print(videogames_df1_final.groupby("productID").size().sort_values(ascending=False)[:10])
#construction of pivot table
#videogames_df1_final.set_index(['userID', 'productID', 'ratings'], append=True)
final_ratings_matrix = videogames_df1_final.pivot(index='userID', columns ='productID', values='ratings').fillna(0)
#print(final_ratings_matrix.head())
print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)

#Density of the rating matrix
num_of_ratings_given = np.count_nonzero(final_ratings_matrix)
print("Number of ratings given: ", num_of_ratings_given)
num_of_ratings_possible = final_ratings_matrix.shape[0]*final_ratings_matrix.shape[1]
print("Possible total of ratings: ", num_of_ratings_possible)
density = (num_of_ratings_given/num_of_ratings_possible)*100
print(f"The density of the matrix is: {density:.2f}%")
# -> Sparse matrix, not many products have ratings from different users

#Split the data randomly into train and test datasets, ratio 70:30
train_data, test_data = train_test_split(videogames_df1_final, test_size = 0.3, random_state=0)
#print(train_data.head())
print('Shape of training data: ',train_data.shape)
print('Shape of testing data: ',test_data.shape)

# ----------------------------------------------------------------------
# Popularity recommender model

#Count of user_id for each unique product as a recommendation score
train_data_grouped = train_data.groupby('productID').agg({'userID': 'count'}).reset_index()
train_data_grouped.rename(columns = {'userID': 'score'}, inplace=True)
#print(train_data_grouped.head(20))

# Sort the products on recommendation score
train_data_sort = train_data_grouped.sort_values(['score', 'productID'], ascending=[0,1])
train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first')
print(train_data_sort.head(5))

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
import sys
#from sklearn.externals import joblib

from scipy.sparse.linalg import svds
import os.path

""" from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings; warnings.simplefilter('ignore')
%matplotlib inline """


def load_data():
    while True:
        filename = input('\033[91mPlease enter the file name: \033[00m')
        filename = filename + '.csv'
        if os.path.isfile(filename) == True:
            break
        else:
            print('\nFile does not exist. Try again')
    columns = ['userID', 'productID', 'ratings', 'timestamp']
    df=pd.read_csv(filename, names=columns)
    df.drop(columns=['timestamp'], inplace=True)
    #rows, columns = videogames_df.shape
    return filename, df

def subset(dataframe, int):
    # take a subset from 0/1 to int
    dataframe_sub=dataframe.iloc[:int]
    return dataframe_sub

def missing_value_check(dataframe):
    print("\nNumber of missing values across columns:\n", dataframe.isnull().sum())
    if dataframe.isnull().sum().sum()==0:
        print("\n-> No missing values")    
    else:
        print("Some values are missing!!")

def plotting(dataframe):
    print("\nHere is the detail count of ratings.")
    print(dataframe['ratings'].value_counts())    
    answer = input("\nWould you like to see a visual representation(plot)?: ")
    if answer == 'y':
        dataframe.groupby('userID')['ratings'].count().sort_values(ascending=False)
        ax = sns.countplot(x='ratings', data=dataframe, palette = 'Set1', edgecolor = 'black')
        for i in range(0,5):
            ax.bar_label(ax.containers[i])
            ax.set_ylabel("Rating score")
            ax.set_ylabel("Number of ratings")
        plt.show()        

def unique_users(dataframe):
    answer = input("\nDo you want to see the number of unique users and products?: ")
    if answer == 'y':
        # Number of unique user id  in the data
        print('\nNumber of unique users in this data subset: ', dataframe['userID'].nunique())
        # Number of unique product id  in the data
        print('Number of unique products in this data subset: ', dataframe['productID'].nunique())  
        input("\nPress enter to continue...")


def top_ten_users(dataframe):
    #Check the top 10 users based on ratings
    most_rated=dataframe.groupby('userID').size().sort_values(ascending=False)[:10]
    print('\nHere is the top 10 users based on the number of ratings given: \n',most_rated)

def top_users_infos(dataframe, numberofratings):
    # print infos about top users selected based on minimum amount of ratings
    print(f"\nNumber of unique users that provided more than {numberofratings} ratings: {dataframe['userID'].nunique()}")
    print(f'Number of unique products rated by these {dataframe['userID'].nunique()} users: : {dataframe['productID'].nunique()}')
    print('Number of total products rated: ', len(dataframe))
    print(f'{len(dataframe)-dataframe['productID'].nunique()} products have been given two or more ratings')
    # checking top 10 products with most ratings
    answer = input("\nDo you want to see the top 10 products based on number ratings?: ")
    if answer == 'y':
        print("\nTop 10 products based on number of ratings from different users:\n")
        print(dataframe.groupby("productID").size().sort_values(ascending=False)[:10])

def matrix_density(dataframe):
    #Density of the rating matrix
    num_of_ratings_given = np.count_nonzero(dataframe)
    print("\nNumber of ratings given: ", num_of_ratings_given)
    num_of_ratings_possible = final_ratings_matrix.shape[0]*final_ratings_matrix.shape[1]
    print("Possible total of ratings: ", num_of_ratings_possible)
    density = (num_of_ratings_given/num_of_ratings_possible)*100
    print(f"The density of the matrix is: {density:.2f}%")
    if density < 66:
        print("\nDensity < 66%, we are dealing with a sparse matrix -> necessity of prediction models")
    else:
        print("\nMatrix not sparse! No further analysis needed :)")
        sys.exit(0)

def popularity_model(dataframe):
    # Count number of ratings from different users (=score) and average rating for each unique product as a recommendation score.
    train_data_grouped = dataframe.groupby('productID').agg({'userID': 'count','ratings': 'mean'}).reset_index()
    train_data_grouped.rename(columns = {'userID': 'number of ratings'}, inplace=True)
    train_data_grouped.rename(columns={'ratings': 'average rating'}, inplace=True)  
    # Sort the products based on the score (total number of ratings). If number of ratings are equal, sort by average
    # rating. Print top 10
    train_data_sort = train_data_grouped.sort_values(['number of ratings', 'average rating'], ascending=False)
    print("\nHere are the top 10 recommended products")
    print(train_data_sort.head(10))
    # Selecting further, deciding of a minimum average rating (will bypass score) and finally the number of recommendations the model proposes
    minimum_rating = float(input("\nSelect a minimum average rating: "))
    number_of_recommendations = int(input("How many recommendations do you want to make?: "))
    popularity_recommendations = train_data_sort.loc[train_data_sort['average rating'] >= minimum_rating]
    popularity_recommendations['rank'] = popularity_recommendations['number of ratings'].rank(ascending=0, method='first') # add of rank column
    print(f'\nHere is your top {number_of_recommendations} recommended products! All users will be recommended these.\n\n', popularity_recommendations.head(number_of_recommendations))
    return popularity_recommendations



def collaborative_filtering_model(matrix, number_of_users):

    # Define user index from 0 to number of user
    matrix['user_index'] = np.arange(matrix.shape[0])
    matrix.set_index(['user_index'], inplace=True)
    #print("Current ratings matrix\n", matrix.head(number_of_users))
    #Lets use matrix factorization to build our model
    # Matrix Factorization using Singular Value Decomposition (SVD). (pivot df dataframe->np.ndarray)

    U, s, V = svds(matrix.to_numpy(), k=number_of_users-1)
    #print("U matrix(users): \n", U, "\n Sigma matrix: \n", s, "\nV matrix (products): \n", V)

    # Sigma is not yet diagonal, so we make it diagonal.
    s = np.diag(s)
    #print('Diagonal matrix: \n',s)

    #Predicted ratings
    all_user_predicted_ratings = np.dot(np.dot(U, s), V) 
    # Convert predicted ratings to a dataframe
    predictions_df = pd.DataFrame(all_user_predicted_ratings, columns = matrix.columns)
    predictions_df['user_index'] = np.arange(predictions_df.shape[0])
    predictions_df.set_index(['user_index'], inplace=True)
    #print(predictions_df.head(number_of_users))
    print("\nYour predictions are ready!")
    return predictions_df

def recommend_products_CF(user, pivot_df, preds_df, num_recommendations):
    sorted_user_ratings = pivot_df.loc[user-1].sort_values(ascending=False)
    sorted_user_predictions = preds_df.loc[user-1].sort_values(ascending=False)
    result = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
    print(result)
    result.index.name = 'Recommended items'
    result.columns = ['user_ratings', 'user_predictions']
    print(result)
    result = result.loc[result.user_ratings==0]
    print(result)
    result = result.sort_values('user_predictions', ascending=False)
    print(f'\nHere are the top {num_recommendations} recommended products for user number: {user}')
    print(result.head(num_recommendations))

def best_prediction(ratings_matrix, prediction_df):
    df = (prediction_df-ratings_matrix).abs()
    df['max'] = df.max(axis=1)
    prediction_df['min'] = prediction_df.min(axis=1)
    print(prediction_df)
    print(df)

# Evaluation of model, Root Mean Squared Error (RMSE)
def RMSE_evalutation(ratings_matrix, prediction_df):
    ratings_matrix.mean() # average  actual rating for each item
    prediction_df.mean() #average predicted rating for each item

    rmse_df = pd.concat([ratings_matrix.mean(), prediction_df.mean()], axis=1)
    rmse_df.columns = ['avg_actual_ratings', 'avg_predicted_ratings']
    rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)


    RMSE = round((((rmse_df.avg_actual_ratings - rmse_df.avg_predicted_ratings) ** 2).mean() ** 0.5), 3)
    print('\nRMSE SVD model = ', RMSE)    

if __name__ == '__main__': 
    # load data from files
    #filename, df = load_data()
    columns = ['userID', 'productID', 'ratings', 'timestamp'] #testing
    df=pd.read_csv('Video_Games.csv', names=columns) #testing
    rows, columns = df.shape
    #print(f"\nThe {filename} data set has {rows} rows (number of ratings) and {columns} columns")
    print(f"\nThe data set has {rows} rows (number of ratings) and {columns} columns")

    #ask user if wants to see quick summary of data
    answer = input("\nDo you want to see the dataframe infos?: ")
    if answer == 'y':
        print('\n\n', df.info())     

    #take a subset (large data amount). (add randomize?)
    subset_number = int(input('\n\033[91mWhat subset would you like to take (number of rows)?: \033[00m'))
    df_sub = subset(df, subset_number)
    del df

 
    #more explicit infos about ratings variable
    answer = input("\nDo you want to see a summary statistics of the ratings variable?: ")
    if answer == 'y':
        print('\n', df_sub['ratings'].describe())
        print(f"Min rating is {df_sub.ratings.min()} and max rating is {df_sub.ratings.max()}")
        input("\nPress enter to continue...")
    #checking for missing values
    print("\nChecking for missing values...")
    missing_value_check(df_sub)

    # plot part
    plotting(df_sub)

    # Checking number of unique users and products
    unique_users(df_sub)

    #Giving the top 10 users based on number of ratings given to get a general idea
    top_ten_users(df_sub)

    # Ask the user if wants to see more users & ratings
    answer = input("\nSee more users and their number of ratings?: ")
    if answer=='y':
        user_number = int(input("\nHow many?: "))
        print(df_sub.groupby('userID').size().sort_values(ascending=False)[:user_number])

    # select the number of top users wanted
    counts = df_sub.userID.value_counts()
    number_of_ratings = int(input("\n\033[91mSet the minimum number of ratings given at: \033[00m"))  
    df_sub_final = df_sub[df_sub.userID.isin(counts[counts>number_of_ratings].index)]
    top_users_infos(df_sub_final, number_of_ratings)
    
    #construction of pivot table. In case of duplicates users/productID (same user giving 2+ rating to a product) we use the mean of ratings(aggfunc) 
    #If a product has no rating from a user, we give it a 0 rating score (fillna(0))
    final_ratings_matrix = df_sub_final.pivot_table(index='userID', columns ='productID', values='ratings', aggfunc='mean').fillna(0)
    print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape, ': (unique users, unique products)')

    # density check of matrix. If d<66% -> sparse matrix
    matrix_density(final_ratings_matrix)

    #Data splitting: split the data randomly into train and test datasets, ratio 70:30. 
    #train_data, test_data = train_test_split(df_sub_final, test_size = 0.3, random_state=0)

    # Choose a prediction model
    answer = input("\nWhat model would you like to make?\n1) Popularity recommender model\n2)Collaborative Filtering recommender model\n")
    while answer != '1' and answer !='2':
        answer = input("Please type 1 or 2 only: ")
    if answer == '1':
        print("\n---------------------------------------------------------------------------------------\nPopularity recommender model")
        popularity_recommendations = popularity_model(df_sub_final)
    if answer == '2':
        print("\n---------------------------------------------------------------------------------------\nCollaborative Filtering recommender model")
        predictions_df = collaborative_filtering_model(final_ratings_matrix, final_ratings_matrix.shape[0])
        #final_ratings_matrix['user_index'] = np.arange(final_ratings_matrix.shape[0])
        #final_ratings_matrix.set_index(['user_index'], inplace=True)
        #print(final_ratings_matrix)
        #print(type(final_ratings_matrix))
        #print(predictions_df)
        #print(type(predictions_df))
        #best_prediction(final_ratings_matrix.head(), predictions_df.head())
        user, num_recommendations = [int(x) for x in input(f"\033[91mEnter a user number (from 1 to {final_ratings_matrix.shape[0]}) and a number of recommendations: \033[00m").split()]
        recommend_products_CF(user, final_ratings_matrix, predictions_df, num_recommendations)

        RMSE_evalutation(final_ratings_matrix, predictions_df)

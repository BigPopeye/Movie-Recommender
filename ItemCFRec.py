#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:23:48 2019

@author: olive
"""

import pandas as pd
from numpy import dot
from numpy.linalg import norm
import datetime
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('../ml-latest-small/ratings.csv', sep=',', usecols=['movieId', 'userId', 'rating'])

ratings = pd.DataFrame(df.groupby('movieId')['rating'].mean())
ratings['number_of_ratings'] = df.groupby('movieId')['rating'].count()

movie_matrix = df.pivot_table(index='movieId', columns='userId', values='rating')

'''
建立innerIId到movieId的映射
'''
inner_matrix = movie_matrix.reset_index()
outerIID = pd.Series(inner_matrix['movieId'],inner_matrix.index)  #innerId -> itemId
innerIID = pd.Series(inner_matrix.index,inner_matrix['movieId'])  #itemId -> innerId
inner_matrix.drop(columns=['movieId'],inplace=True)

# normalization ： min-max norm
movie_matrix_norm = (inner_matrix - inner_matrix.min()) / (inner_matrix.max() - inner_matrix.min())



#init target user data

targetUser = {'userId': [611, 611, 611, 611, 611],
              'movieId': [1, 32, 215, 507, 648],
              'rating': [5, 5, 4, 4, 3]}

target_df = pd.DataFrame.from_dict(targetUser)

'''
Calculate cosine similarity
'''
simMovieRatings = pd.DataFrame(index=movie_matrix_norm.index, columns=movie_matrix_norm.index)

def Cosine(x, y):
    return dot(x,y) / (norm(x) * norm(y))

start_stamp = datetime.datetime.now()
print("start_time       " + start_stamp.strftime('%Y.%m.%d-%H:%M:%S') )

#Method 1 ：manually calculate cosine, very slow,only need to calculate ralated movie
#for i in targetUser['movieId']: 
#    iid = innerIID[i]
#    for j in range(0, len(simMovieRatings.columns)):
#        simMovieRatings.at[j,iid] = Cosine(movie_matrix_norm.iloc[iid].fillna(0), movie_matrix_norm.iloc[j].fillna(0))


#Method 2 ： sklearn cosine function, very fast, can be used to generate full cosine matrix
simMovieRatings = pd.DataFrame(cosine_similarity(movie_matrix_norm.fillna(0)))

end_stamp = datetime.datetime.now()
print("end_time       " + end_stamp.strftime('%Y.%m.%d-%H:%M:%S') )

'''
Rank by similarity * score
'''
absi = 0 #绝对位置
sim_score_topN_raw = pd.Series()
for i in targetUser['movieId']:
    inner_id = innerIID[i]
    
    target_score = target_df.iloc[absi,2]
    sim_rank = simMovieRatings.sort_values(by=simMovieRatings.columns[inner_id],ascending=False).iloc[:,[inner_id]].head(100)
    sim_score = sim_rank.iloc[:,0] * target_score
    sim_score_topN_raw = sim_score_topN_raw.append(sim_score) #将每个电影的相似电影汇总到一起
    absi = absi+1

sim_score_topN = sim_score_topN_raw.groupby(by = sim_score_topN_raw.index).max()  #对相同的电影，评分选最大的，然后
sim_score_topN.sort_values(ascending=False,inplace=True)
topN = pd.DataFrame(outerIID.filter(items=sim_score_topN.index)) #将innerIId转为itemId,转为DataFrame方便后续操作
topN_with_score = topN.join(sim_score_topN.rename('score'))

'''
Output recommendation
'''
movies = pd.read_csv('../ml-latest-small/movies.csv', sep=',', usecols=['movieId', 'title', 'genres']) 
topN_movies = topN_with_score.set_index('movieId').join(movies.set_index('movieId'))
topN_movies_rec=topN_movies.drop(topN_movies.index[:5]).head(10)
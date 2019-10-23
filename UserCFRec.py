#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:31:31 2019

@author: olive
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:23:48 2019

@author: olive
"""

import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import datetime

# 读取用户对电影评分的数据rating.csv
df = pd.read_csv('../ml-latest-small/ratings.csv', sep=',', usecols=['movieId', 'userId', 'rating'])


user_matrix = df.pivot_table(index='userId',columns='movieId',values='rating')
#对评分进行归一化 ： min-max 归一化
user_matrix_norm = (user_matrix - user_matrix.min()) / (user_matrix.max() - user_matrix.min())

#'''
#初始化待推荐用户的偏好数组————itembased 推荐不需要更新matrix，user-based推荐需要更新
#1. 用户ID自增1
#2. 已有评分的数据写入对应item项，其他项置为0
#3. 将数组归一化
#4. 将数组插入原data frame
#'''
targetUID = 611;
## TBD 改为自动获取targetID
targetUser = {'userId': [611, 611, 611, 611, 611],
              'movieId': [1, 32, 215, 507, 648],
              'rating': [5, 5, 4, 4, 3]}

target_df = pd.DataFrame.from_dict(targetUser)
target_matrixT = target_df.pivot_table(index='movieId', columns='userId', values='rating')
target_matrix_normT = (target_matrixT - target_matrixT.min()) / (target_matrixT.max() - target_matrixT.min())
target_matrix_norm = target_matrix_normT.T

user_vector = pd.Series(index=user_matrix_norm.columns) # add target user into matrix
for j in targetUser['movieId']:
    user_vector.at[j] = target_matrix_norm.at[targetUID,j]


'''
Calculate cosine similarity
'''
def Cosine(x, y):
    return dot(x,y) / (norm(x) * norm(y))

start_stamp = datetime.datetime.now()
print("start_time       " + start_stamp.strftime('%Y.%m.%d-%H:%M:%S') )

#Method 1 ：manually calculate cosine, very slow,only need to calculate this users with other else

def SimilarUsers(simUserDF, userVector): # get similar users similarity  matrix
    simMatrix = pd.DataFrame(columns=simUserDF.index)
    for j in range(1, len(user_matrix_norm.index)):
        simMatrix.at[0,j] = Cosine(userVector.T.fillna(0), simUserDF.loc[j].fillna(0))  #改为单个向量和DF相乘
    return simMatrix.T

def TopNSimUsers(simSeries, n): # get top N similar userss
    return simSeries.sort_values(by=simSeries.columns[0],ascending=False).iloc[:,0].head(n)
    
    
simMatrix = SimilarUsers(user_matrix_norm, user_vector)
topNSimUsers = TopNSimUsers(simMatrix, 50)


def SimilarScores(simUsers,userMatrix): # get similar series by multiply similarity and ratings
    simScores = pd.Series()
    for i in simUsers.index:
        movies = userMatrix.loc[i].dropna().T
        for j in movies.index:
            temp = pd.Series()
            temp.at[j] = simUsers.at[i] * userMatrix.loc[i,j]
            simScores = simScores.append(temp)
    return simScores

def topN_simItems(simScores,n):   
#    tempdf = pd.DataFrame(data=simScores)
    tempdf = simScores.reset_index()
    simScores_topN = tempdf.groupby(by =tempdf.columns[0]).max() 
    return simScores_topN.sort_values(by=simScores_topN.columns[0],ascending=False).head(n)

simScores =  SimilarScores(topNSimUsers, user_matrix)      
topNMovies = topN_simItems(simScores,20)

def itemId2Names(items,dicts,columnName):
    return items.join(dicts.set_index(columnName))

movies = pd.read_csv('../ml-latest-small/movies.csv', sep=',', usecols=['movieId', 'title', 'genres']) 
colName='movieId'
top_movies = itemId2Names(topNMovies,movies,colName)
    
def RecMovies(items,n,watched): # get recommendation by select topN high score movies and drop thoes have already been seen
    rec_items = items
#    rec_items['rec'] = rec_items.apply(lambda x: np.nan if x.index is in watched.index else 1, axis=1)
    for i in items.index:
        if i in watched.values:
            rec_items.at[i,'rec'] = np.nan
        else:
            rec_items.at[i,'rec'] = 1
    return rec_items.dropna().head(n)
    
recMovies = RecMovies(top_movies,10,target_df['movieId']) 
    

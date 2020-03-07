import os
import pandas as pd
from sqlalchemy.orm import mapper

df_tasks = pd.read_csv("Tasks.csv", names=['taskId2','Reason'], dtype = {'taskId2': 'int32', 'Reason':'str'})

df_ratings = pd.read_csv("DataVisitor3.csv",names=['id','taskId','user','Acc','HR','Man','Sec','Kitchen','Toilets','Gamesroom','Rating','TaskName','taskId2'], dtype={'id':'int32','taskId':'int32','Rating':'int32','TaskName':'str','taskId2':'int32'})

#print(df_tasks.head())
df_tasks.shape
#print(df_ratings.head())

df_ratings=df_ratings[:100]
df_ratings.shape

from scipy.sparse import csr_matrix

df_task_features = df_ratings.pivot(values='Rating', index='id', columns='taskId2').fillna(0)

mat_task_features = csr_matrix(df_task_features.values)

#print(df_task_features.head())

from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=4, n_jobs=-1)


num_users = len(df_ratings.id.unique())
num_tasks = len(df_ratings.taskId2.unique())

#print('There are {} unique users and {} unique task in this data set'.format(num_users, num_tasks))

# get count
df_ratings_cnt_tmp = pd.DataFrame(df_ratings.groupby('Rating').size(), columns=['count'])



total_cnt = num_users * num_tasks
rating_zero_count = total_cnt - df_ratings.shape[0]

df_ratings_count = df_ratings_cnt_tmp.append(
    pd.DataFrame({'count':rating_zero_count}, index=[0.0]),
    verify_integrity=False,
).sort_index()

#print(df_ratings_count)

#log normalise to make it easier to interpret on a graph
import numpy as np
df_ratings_count['log_count'] = np.where(df_ratings_count['count'] > 0.0000000001,np.log(df_ratings_count['count']),10)
#print(df_ratings_count)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

ax = df_ratings_count[['count']].reset_index().rename(columns={'index': 'rating score'}).plot(
    x='rating score',
    y='count',
    kind='bar',
    figsize=(12, 8),
    title='Count for Each Rating Score (in Log Scale)',
    logy=True,
    fontsize=12,
)
ax.set_xlabel("Task rating score")
ax.set_ylabel("number of ratings")

# get rating frequency
#number of ratings each movie got.
df_tasks_cnt = pd.DataFrame(df_ratings.groupby('taskId2').size(), columns=['count'])
#print(df_tasks_cnt.head())

popularity_thres = 5
popular_tasks = list(set(df_tasks_cnt.query('count >= @popularity_thres').index))

df_ratings_drop_tasks = df_ratings[df_ratings.taskId2.isin(popular_tasks)]
#print(df_ratings_drop_tasks)
#print('shape of original ratings data: ', df_ratings.shape)
#print('shape of ratings data after dropping unpopular tasks: ', df_ratings_drop_tasks.shape)

# get number of ratings given by every user
df_users_cnt = pd.DataFrame(df_ratings_drop_tasks.groupby('taskId2').size(), columns=['count'])
print("here:",df_users_cnt.head())

rating_thres = 3
active_users = list(set(df_users_cnt.query('count >= @rating_thres').index))
df_ratings_drop_users = df_ratings_drop_tasks[df_ratings_drop_tasks.id.isin(active_users)]
#print('shape of original ratings data: ', df_ratings.shape)
#print('shape of ratings data after dropping both unpopular tasks and inactive users: ', df_ratings_drop_users.shape)

print(df_ratings_drop_users.head())
task_user_mat = df_ratings_drop_users.pivot(index='id', columns='taskId2', values='Rating').fillna(0)

task_to_idx = {
    task: i for i, task in enumerate(list(df_tasks.set_index('taskId2').loc[task_user_mat.index].Reason))
}

task_user_mat_sparse = csr_matrix(task_user_mat.values)

print(task_user_mat)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute',n_neighbors=3, n_jobs=-1)
print(task_user_mat_sparse)
model_knn.fit(task_user_mat_sparse)

import fuzzywuzzy.fuzz as fuzz

def fuzzy_matching(mapper, fav_task, verbose=True):
    match_tuple = []

    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_task.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))

    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        #print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]

def make_recommendation(model_knn, data, mapper, fav_task, n_recommendationsIn):
    model_knn.fit(data)

    print('You have input task:', fav_task)
    idx = fuzzy_matching(mapper, fav_task, verbose=False)

    print('Recommendation system start to make inference')
    print('..........\n')
    distance, indexes = model_knn.kneighbors(data[idx],n_neighbors=n_recommendationsIn)

    raw_recommends = sorted(list(zip(indexes.squeeze().tolist(), distance.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

    reverse_mapper = {v: k for k, v in mapper.items()}

    print('Recommendations for {}:'.format(fav_task))
    for i, (idx,dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distances of {2}'.format(i+1, reverse_mapper[idx], dist))

my_favourite = '1000'

make_recommendation(model_knn=model_knn, data=task_user_mat_sparse, fav_task=my_favourite, mapper=task_to_idx, n_recommendationsIn=3)

print(task_to_idx)
import os
import time
import gc
import argparse
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

class KnnRecommenderSystemVisit:
    """
    Item-based collaborative filtering recommender with KNN implemented using sklearn
    """
    def __init__(self, path_tasks, path_ratings):
        self.path_tasks = path_tasks
        self.path_ratings = path_ratings
        self.task_rating_thres = 0
        self.user_rating_thres = 0
        self.model = NearestNeighbors()

    def set_filter_params(self, task_rating_thres, user_rating_thres):
        """
        Set rating frequency threshold to filter unpopular tasks

        Parameters
        ----------
        task_rating_thres: int, minimum of ratings given by users
        """
        self.task_rating_thres = task_rating_thres
        self.user_rating_thres = user_rating_thres

    def set_model_params(self, n_neighbors, algorithm, metric, n_jobs=None):
        """
        n_neighbors: int, optional (default = 5)

        algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional

        metric: string or callable, default 'minkowski'

        n_jobs: int or None, optional (default=None)
        """

        self.model.set_params(**{
            'n_neighbors': n_neighbors,
            'algorithm': algorithm,
            'metric': metric,
            'n_jobs': n_jobs})

    def _prep_data(self):
        """
        prepare data for recommender

        task-user scipy sparse matrix
        hashmap of task to row index in task-user scipy sparse matrix
        """
        columns = ['id', 'id2', 'Task', 'TaskId', 'rating']
        columns2 = ['TaskId', 'TaskName', 'Task']
        df_tasks = pd.read_csv('DataRecommenderSystem/TaskVisitVisitor.csv', sep=',', names = columns2)
        df_ratings = pd.read_csv('DataRecommenderSystem/DataVisitVisitor.csv',sep=',', names=columns)
        # filter data
        df_tasks_cnt = pd.DataFrame(
            df_ratings.groupby('TaskId').size(),
            columns=['count'])
        popular_tasks = list(set(df_tasks_cnt.query('count >= @self.task_rating_thres').index))  # noqa
        tasks_filter = df_ratings.TaskId.isin(popular_tasks).values

        df_users_cnt = pd.DataFrame(
            df_ratings.groupby('id').size(),columns=['count'])
        active_users = list(set(df_users_cnt.query('count >= @self.user_rating_thres').index))  # noqa
        users_filter = df_ratings.id.isin(active_users).values
        task_names = df_tasks[['TaskId', 'TaskName', 'Task']]
        combined_task_data = pd.merge(df_ratings, task_names, on='TaskId')
        combined_task_data.head()
        combined_task_data.groupby('TaskId')['rating'].count().sort_values(ascending=True).head()

        filter_interview = combined_task_data['TaskId'] == 4
        combined_task_data[filter_interview]['TaskName'].unique()

        # pivot and create movie-user matrix
        tasks_user_mat = combined_task_data.pivot(
            values='rating', index='id', columns='TaskName').fillna(0.1)
        # create mapper from movie title to index
        hashmap = {
            TaskName: i for i, TaskName in
            enumerate(list(df_tasks.set_index('TaskId').loc[tasks_user_mat.index].TaskName))  # noqa
        }
        from math import isnan
        hashmap = {key:val for key, val in hashmap.items() if val != 99}
        # transform matrix to scipy sparse matrix
        tasks_user_mat_sparse = csr_matrix(tasks_user_mat.values)

        # clean up
        del df_tasks, df_tasks_cnt, df_users_cnt
        del df_ratings, filter_interview, tasks_user_mat
        gc.collect()
        return tasks_user_mat_sparse, hashmap

    def _fuzzy_matching(self, hashmap, fav_task):
        """
        return the closest match via fuzzy ratio.
        If no match found, return None
        Parameters
        ----------
        hashmap: dict, map task name to id of task
        fav_task: str, name of user input movie
        Return
        ------
        index of the closest match
        """
        match_tuple = []
        # get match
        for TaskName, idx in hashmap.items():
            if idx == 98:
                print()
            else:
                ratio = fuzz.ratio(TaskName.lower(), fav_task.lower())
                if ratio >= 60:
                    match_tuple.append((TaskName, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
        else:
            print('Found possible matches in our database: '
                  '{0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]

    def _inference(self, model, data, hashmap,
                   fav_task, n_recommendations):
        """
        return top n similar movie recommendations based on user's chosen task
        Parameters
        ----------
        model: sklearn model, knn model
        data: task-user matrix
        hashmap: dict, map movie title name to index of the movie in data
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations
        Return
        ------
        list of top n similar movie recommendations
        """
        # fit
        model.fit(data)
        # get input movie index
        print('Inputted movie: ', fav_task)
        idx = self._fuzzy_matching(hashmap, fav_task)
        # inference
        print('Recommendation system start to make inference')
        print('......\n')
        t0 = time.time()
        distances, indices = model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations + 1)
        # get list of raw idx of recommendations
        raw_recommends = \
            sorted(
                list(
                    zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist()
                    )
                ),
                key=lambda x: x[1]
            )[:0:-1]
        print('It took my system {:.2f}s to make inference \n\
              '.format(time.time() - t0))
        # return recommendation (taskId, distance)
        return raw_recommends

    def make_recommendations(self, fav_task, n_recommendations):
        """
        make top n task recommendations
        Parameters
        ----------
        fav_task: str, name of user input movie
        n_recommendations: int, top n recommendations
        """
        # get data
        task_user_mat_sparse, hashmap = self._prep_data()
        # get recommendations
        raw_recommends = self._inference(
            self.model, task_user_mat_sparse, hashmap,
            fav_task, n_recommendations)
        # print results
        reverse_hashmap = {v: k for k, v in hashmap.items()}
        print('Recommendations for {}:'.format(fav_task))
        for i, (idx, dist) in enumerate(raw_recommends):
            if idx == 98:
                print()
            else:
                print(idx,dist)

if __name__ == '__main__':

    # initial recommender system
    recommender = KnnRecommenderSystemVisit(
        "","")
    # set params
    recommender.set_filter_params(0.7, 0.7)
    recommender.set_model_params(11, 'brute', 'cosine', -1)
    # make recommendations
    recommender.make_recommendations("Task1", 10)
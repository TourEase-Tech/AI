import pandas as pd
import numpy as np
import math
from connection import connect_to_database
from ContentBased import ContentBased
from sortedcontainers import SortedList

content_based = ContentBased()
class MatrixFactorizationRecommenderSystem:
    def __init__(self):
        self.contentBasedModel = content_based
        self.tours = content_based.tours
        self.likes = self.load_data_likes_from_mongodb()
        self.user_to_tour = {}
        self.tour_to_user = {}
        self.tour_user_liked = {}
        self.recommend_history = {}
        self.k = 1
        
        for row in self.tours.values():
            id = row['_id']
            self.user_to_tour[id] = []

        for row in self.likes.values():
            tour_id = row['tour_id']
            user_id = row['user_id']
            liked = row['liked']
            self.user_to_tour.setdefault(user_id, []).append(tour_id)

            self.tour_to_user.setdefault(tour_id, []).append(user_id)

            self.tour_user_liked[(tour_id, user_id)] = liked
        
        self.W = dict.fromkeys(self.user_to_tour.keys())
        self.U = dict.fromkeys(self.tour_to_user.keys())

        self.saved_W = self.W.copy()
        self.saved_U = self.U.copy()
    def load_data_likes_from_mongodb(self):
        likes = {}
        try:
            connect = connect_to_database()
            db = connect['test']
            collection = db['likes']
            data_like = collection.find({})
            for data in data_like:
                like_id = str(data['_id'])
                user_id = str(data['user'])
                tour_id = str(data['tour'])
                like_info = {
                    'user_id': user_id,
                    'tour_id': tour_id,
                    'liked': 1
                }
                likes[like_id] = like_info
        except Exception as e:
            print(f"An error occurred cai ss: {str(e)}")
        return likes
    
    # reset data
    def dump(self):
        self.likes = self.load_data_likes_from_mongodb()
        self.user_to_tour = {}
        self.tour_to_user = {}
        self.tour_user_liked = {}

        for row in self.tours.values():
            id = row['_id']
            self.user_to_tour[id] = []
        
        for row in self.likes.values():
            tour_id = row['tour_id']
            user_id = row['user_id']
            liked = row['liked']
            self.user_to_tour.setdefault(user_id, []).append(tour_id)

            self.tour_to_user.setdefault(tour_id, []).append(user_id)

            self.tour_user_liked[(tour_id, user_id)] = liked

        
        
        for user_id in self.W.keys():
            self.W[user_id] = np.random.randn(self.k)
        for tour_id in self.U.keys():
            self.U[tour_id] = np.random.randn(self.k)

        self.saved_W = self.W.copy()
        self.saved_U = self.U.copy()
        
    def fit(self, epoch, learning_rate, weight):
        self.dump()

        for i in range(0,epoch):
            loss = 0

            for tour_id, user_id in self.tour_user_liked:
                liked = self.tour_user_liked[(tour_id,user_id)]

                predict_liked = np.dot(self.W[user_id], self.U[tour_id])

                error = liked - predict_liked

                saved_W = np.copy(self.W[user_id])
                saved_U = np.copy(self.U[tour_id])

                self.W[user_id] += learning_rate * (2 * error * saved_U - weight * saved_W)
                self.U[tour_id] += learning_rate * (2 * error * saved_W - weight * saved_U)

                loss += error ** 2
            
            loss = loss / len(self.tour_user_liked)
            rmse = math.sqrt(loss)
            print(f'Epoch: {i + 1}/{epoch}')
            print('Loss: ' , loss, 'RMSE: ', rmse)
        

        self.saved_W = self.W
        self.saved_U = self.U

        self.recommend_history = {}
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def get_rating(self, user_id, tour_id):
        return self.sigmoid(np.dot(self.saved_W[user_id], self.saved_U[tour_id]))
    
    def recommend_matrix_factorization(self, user_id, n = 10):
        recommend_list = SortedList()

        for tour_id in self.tour_to_user.keys():
            if(tour_id, user_id) not in self.tour_user_liked:
                liked = self.get_rating(user_id, tour_id)
                sim = []
                for tour_id_user_liked in self.user_to_tour[user_id]:
                    sim.append(self.contentBasedModel.get_tour_similarities(tour_id_user_liked, tour_id))
                
                adjust = 0

                if(user_id, tour_id) in self.recommend_history:
                    # Hao tru 0.3 moi lan
                    adjust = self.recommend_history[(user_id,tour_id)] * 0.03

                recommend_list.add((np.mean(sim) * liked - adjust, tour_id))
                if len(recommend_list) > n:
                    del recommend_list[0]
        result = []
        for score, tour_id in list(recommend_list):
            if(user_id, tour_id) in self.recommend_history:
                self.recommend_history[(user_id,tour_id)] += 1
            else:
                self.recommend_history[(user_id,tour_id)] = 0
            result.append({
                'id': tour_id,
                'name': self.tours[tour_id]['name'],
                'description': self.tours[tour_id]['description'],
                'price': self.tours[tour_id]['price'],
                'period': self.tours[tour_id]['period'],
                'score': round(score,2)
            })
        return result

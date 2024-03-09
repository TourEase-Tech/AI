from sortedcontainers import SortedList
from datetime import datetime
from connection import connect_to_database
import heapq
import re

class ContentBased:
    def __init__(self):
        self.tours = self.load_data_from_mongodb()

    def load_data_from_mongodb(self):
        tours = {}
        try:
            connect = connect_to_database()
            db = connect['test']
            collection = db['tours']
            data_tour = collection.find({})
            for data in data_tour:
                tour_id = str(data['_id'])
                period = self.convert_to_hours(data['period'])
                tour = {
                    '_id': tour_id,
                    'name': data['name'],
                    'description': data['description'],
                    'price': int(data['price']), 
                    'departureLocation': data['departureLocation'],
                    'period': period,
                    'images': data['images'],
                    'destination': data['destination'],
                    'departureDay': data['departureDay'],
                }
                tours.update({tour_id: tour})
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        return tours

    def convert_to_hours(self, period_string):
        try:
            days = int(period_string.split('ngày')[0])
        except:
            days = 0
        try:
            nights = int(period_string.split('đêm')[0])
        except:
            nights = 0
        total_hours = (days + nights) * 12
        return total_hours
    
    def get_sim_name(self, tour_i, tour_j):
        name_i = self.tours[tour_i]['name']
        name_j = self.tours[tour_j]['name']

        tokens_i = re.findall(r'\w+', name_i.lower())  
        tokens_j = re.findall(r'\w+', name_j.lower())
        common_tokens = set(tokens_i) & set(tokens_j)
        similarity = len(common_tokens) / len(name_i)

        return similarity
        
    def get_sim_departureLocation(self,tour_id_i, tour_id_j):
        departureLocation_i = set(self.tours[tour_id_i]['departureLocation'])
        departureLocation_j = set(self.tours[tour_id_j]['departureLocation'])
        similarity = len(departureLocation_i.intersection(departureLocation_j))/len(departureLocation_i.union(departureLocation_j))

        return similarity 

    def get_sim_price(self, tour_i, tour_j):
        price_i = self.tours[tour_i]['price']
        price_j = self.tours[tour_j]['price']
        if price_i == 0 or price_j == 0:
            return 0
        max_price = max(price_i, price_j)

        normalized_price_i = price_i / max_price
        normalized_price_j = price_j / max_price

        normalized_price_i = (max_price- price_i ) / (max_price )
        normalized_price_j = (max_price - price_j) / (max_price )

        similarity = abs(normalized_price_i - normalized_price_j)

        return similarity 

    def get_sim_period(self, tour_i, tour_j):
        period_i = float(self.tours[tour_i]['period'])
        period_j = float(self.tours[tour_j]['period'])
        diff = abs(period_i - period_j)

        if diff < 12:
            return 1
        elif diff < 24:
            return 0.8
        elif diff < 48:
            return 0.6
        return 0

    def get_sim_departureDay(self, tour_i, tour_j):
        departure_day_i = self.tours[tour_i]['departureDay']
        departure_day_j = self.tours[tour_j]['departureDay']
        similarity = 1 / (1 + abs((departure_day_i - departure_day_j).days))

        return similarity 


    def get_tour_similarities(self, tour_i, tour_j):
        sim_name = self.get_sim_name(tour_i, tour_j)
        sim_departureLocation = self.get_sim_departureLocation(tour_i, tour_j)
        sim_price = self.get_sim_price(tour_i, tour_j)
        sim_period = self.get_sim_period(tour_i, tour_j)
        sim_departureDay = self.get_sim_departureDay(tour_i, tour_j)

        return 0.2 * (sim_name + sim_price + sim_period + sim_departureDay + sim_departureLocation)
    
    def recommend(self, tour_id):
        k = 10
        list = SortedList()

        for id in self.tours:
            if id == tour_id:
                continue
            sim = self.get_tour_similarities(tour_id, id)
            list.add((sim,id))
            if len(list) > k:
               del list[0]
        result = []  
        for score, id in list:
            result.append({
                'id':id,
                'name': self.tours[id]['name'],
                'description': self.tours[id]['description'],
                'price': self.tours[id]['price'],
                'departureLocation': self.tours[id]['departureLocation'],
                'period': self.tours[id]['period'],
                'images': self.tours[id]['images'],
                'destination': self.tours[id]['destination'],
                'departureDay': self.tours[id]['departureDay'],
                'score': round(score,2)
            })
        return result

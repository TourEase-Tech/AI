from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGODB_CONNECTION_STRING = os.getenv('DATABASE').replace('<password>', os.getenv('DATABASE_PASSWORD'))

def connect_to_database():
    connect = MongoClient(MONGODB_CONNECTION_STRING)
    return connect

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'train_data.csv')
TEST_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'test_data.csv')
MODEL_NAME = 'trained_model.h5'
OUTPUT_DIR = os.path.join(BASE_DIR, 'models')

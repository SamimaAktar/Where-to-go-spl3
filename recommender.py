import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from collections import defaultdict

header = ['user_id', 'location_id', 'frequency']
df = pd.read_csv('../gowalla/preprocessed_data.csv', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_locations = df.location_id.unique().shape[0]
print('>> Number of users = ' + str(n_users))
print('>> Number of locations = ' + str(n_locations))


#split data train and test
train_data, test_data = ms.train_test_split(df, test_size=0.20)
print(">> Train data rows and columns:", train_data.shape)
print(">> Test data rows and columns:", test_data.shape)

print(df.head())

train_data_matrix = np.zeros((n_users, n_locations))
for checkin in train_data.itertuples():
    train_data_matrix[checkin[1], checkin[2]] = checkin[3]

# for RMSE
test_data_matrix = np.zeros((n_users, n_locations))
# for Precision and Recall
ground_truth_dic = defaultdict(set)

for checkin in test_data.itertuples():
    test_data_matrix[checkin[1], checkin[2]] = checkin[3]
    ground_truth_dic[int(checkin[1])].add(int(checkin[2]))

print(train_data.head())

def cosine_similarity(train_matrix, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = train_matrix.dot(train_matrix.T) + epsilon
    elif kind == 'location':
        sim = train_matrix.T.dot(train_matrix) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

user_similarity = cosine_similarity(train_data_matrix, kind='user')
item_similarity = cosine_similarity(train_data_matrix, kind='location')
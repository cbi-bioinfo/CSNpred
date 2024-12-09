import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import pairwise_distances
import sys

train_x_filename = sys.argv[1]
train_y_filename = sys.argv[2]
test_x_filename = sys.argv[3]
num_neighbor = int(sys.argv[4])

resDir = "./CSN_data"
os.makedirs(resDir, exist_ok = True)

X_train = pd.read_csv(train_x_filename)
y_train = pd.read_csv(train_y_filename)
X_test = pd.read_csv(test_x_filename)

top_sample_idx_list = []
for i in range(len(X_train)) :
	tmp = pairwise_distances(pd.DataFrame(X_train.iloc[i]).T, X_train)[0]
	top_sample_idx_list.append(tmp.argsort()[1:num_neighbor+1].tolist())

del X_train['array_row']
del X_train['array_col']

x_train_np_data = []
top_sample_idx_list = pd.DataFrame(top_sample_idx_list)
for i in range(len(top_sample_idx_list)) :
	sample_list = [i]
	sample_list.extend(top_sample_idx_list.iloc[i].tolist())
	x_train_np_data.append(X_train.iloc[sample_list].values)

x_train_np_data = np.array(x_train_np_data)
np.save(os.path.join(resDir, "gcn_neigh_train_X.npy"), x_train_np_data)

adj_df = []
for j in range(len(X_train)) :
	tmp_df = np.zeros((num_neighbor + 1, num_neighbor + 1))
	tmp_df = pd.DataFrame(tmp_df)
	tmp_list = [0]
	for i in range(num_neighbor) :
		tmp_list.append(1)
	tmp_df[0] = tmp_list
	tmp_df.iloc[0] = tmp_list
	tmp_df = tmp_df.astype('int')
	adj_df.append(tmp_df)

adj_array = np.array(adj_df)
np.save(os.path.join(resDir, "adj_mat_neigh_train_X.npy"), adj_array)

# Test
top_sample_idx_list = []
for i in range(len(X_test)) :
	tmp = pairwise_distances(pd.DataFrame(X_test.iloc[i]).T, X_test)[0]
	top_sample_idx_list.append(tmp.argsort()[1:num_neighbor+1].tolist())

del X_test['array_row']
del X_test['array_col']

x_test_np_data = []
top_sample_idx_list = pd.DataFrame(top_sample_idx_list)
for i in range(len(top_sample_idx_list)) :
	sample_list = [i]
	sample_list.extend(top_sample_idx_list.iloc[i].tolist())
	x_test_np_data.append(X_test.iloc[sample_list].values)

x_test_np_data = np.array(x_test_np_data)
np.save(os.path.join(resDir, "gcn_neigh_test_X.npy"), x_test_np_data)


adj_df = []
for j in range(len(X_test)) :
	tmp_df = np.zeros((num_neighbor + 1, num_neighbor + 1))
	tmp_df = pd.DataFrame(tmp_df)
	tmp_list = [0]
	for i in range(num_neighbor) :
		tmp_list.append(1)
	tmp_df[0] = tmp_list
	tmp_df.iloc[0] = tmp_list
	tmp_df = tmp_df.astype('int')
	adj_df.append(tmp_df)

adj_array = np.array(adj_df)
np.save(os.path.join(resDir, "adj_mat_neigh_test_X.npy"), adj_array)



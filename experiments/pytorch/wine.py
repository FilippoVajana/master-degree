#%% [markdown]
# # Handmade classification of wines

#%%
# init
import csv
import urllib.request
import numpy as np
import torch


#%%
# download dataset
print('Downloading dataset')
ds_url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
ds_path = r"./data/winequality-white.csv"
urllib.request.urlretrieve(ds_url, ds_path)


#%%
# load dataset
wineq_numpy = np.loadtxt(ds_path, dtype=np.float32, delimiter=";", skiprows=1)
wineq_numpy
# check if data have been loaded
wine_file = open(ds_path)
col_list = next(csv.reader(wine_file, delimiter=';'))
wineq_numpy.shape, col_list

#%%
# convert to tensor
wineq = torch.from_numpy(wineq_numpy)
wineq.shape, wineq.type()


#%%
# split dataset
data = wineq[:, :-1]
target = wineq[:, -1]
print("data:", data)
print("target:", target)

#%%
# one-hot encoding
def onehot_encoding(indexing_tensor):
    one_tensor = torch.zeros(indexing_tensor.shape[0], 10) # (NumLabels,10) dim
    onehot = one_tensor.scatter(1, indexing_tensor.unsqueeze(1).long(), 1.0)
    # print(onehot)
    return onehot


#%%
# some analysis on data
data_mean = torch.mean(data, dim=0)
data_var = torch.var(data, dim=0)
data_stdev = torch.std(data, dim=0)
data_normalized = (data - data_mean) / data_stdev
data_normalized

#%%
# group data in quality classes
bad_data = data[torch.le(target, 3)]
mid_data = data[torch.gt(target, 3) & torch.lt(target, 7)]
good_data = data[torch.ge(target, 7)]

bad_mean = bad_data.mean(dim=0)
mid_mean = mid_data.mean(dim=0)
good_mean = good_data.mean(dim=0)

def print_group_value(col_list, bad_mean, mid_mean, good_mean):
    for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
        print('{:2}  {:20}  {:6.2f}  {:6.2f}  {:6.2f}'.format(i, *args))

print_group_value(col_list, bad_mean, mid_mean, good_mean)

#%%
# seems like that the total sulfur dioxide is a good indicator of quality
total_sulfur_threshold = mid_mean[6]
total_sulfur_data = data[:, 6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
print("predicted good:", predicted_indexes.sum())

#%%
# now get indexes of good wines and compare with the prediction
good_indexes = torch.gt(target, 5)
print("actual good:", good_indexes.sum())

n_matches = torch.sum(predicted_indexes & good_indexes).item()
n_predicted = predicted_indexes.sum().item()
n_good = good_indexes.sum().item()

print('matches: {:.2f}    predicted: {:.2f}    good: {:.2f}'.format(n_matches, n_matches/n_predicted, n_matches/n_good))
# only 61% of actual good wines will be identified using the sulfur data only
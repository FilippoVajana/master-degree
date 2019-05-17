#%% [markdown]
# # Time series analysis on bike sharing data

#%%
# init
import csv
import datetime
import numpy as np
import torch

#%%
# reading dataset
def month_day(date):
    fmt = r'%Y-%m-%d'
    dt = datetime.datetime.strptime(date.decode("utf-8"), fmt)
    tt = dt.timetuple()
    return tt.tm_mday

# load dataset
path = r"./data/ts_bikes/hour-fixed.csv"
bikes_numpy = np.loadtxt(path,
                        dtype=np.float32,
                        delimiter=',',
                        skiprows=1,
                        converters={1: lambda x: month_day(x)})

bikes = torch.from_numpy(bikes_numpy)
bikes

with open(path, "r") as f:
    reader = csv.reader(f)
    h = next(reader)
    bikes_cols = h

#%%
# split data day by day
daily_bikes = bikes.view(-1, 24, bikes.shape[1])

# transpose to (sample, channels, hours)
print(daily_bikes.shape)
daily_bikes.transpose_(1,2)
print(daily_bikes.shape)

#%%
# encode weather data
daily_weather_onehot = torch.zeros(daily_bikes.shape[0],
                                    4,
                                    daily_bikes.shape[2])

daily_weather_onehot.scatter_(1, daily_bikes[:,9,:].long().unsqueeze(1) - 1, 1.0)
daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)
print(daily_bikes)

#%%
# normalize
def normalize(tensor):
    min = tensor.min()
    max = tensor.max()
    tensor_norm = (tensor - min) / (max - min) # [0,1] interval
    return tensor_norm

print(normalize(daily_bikes[:,10,:]))

#%%
# one-hot encoding season data
season = bikes[:,2]

season_onehot = torch.zeros(season.shape[0], 4)

print("season:", season.shape)
print("onehot:", season_onehot.shape)
season_onehot.scatter_(1, season.long().unsqueeze(1) - 1, 1.0)
print(season_onehot)

#%%
# add one-hot to data
bikes_mod = bikes.clone()
bikes_mod = torch.cat((bikes_mod, season_onehot), dim=1)

print(bikes_mod[1])

#%%
x = torch.tensor([1,2,3,4])
print(x, x.shape)

y = x.unsqueeze(0)
print(y, y.shape)

z = x.unsqueeze(1)
print(z, z.shape)


#%%

import nflreadpy as nfl
import pandas as pd
import xgboost as xgb
pbp = nfl.load_pbp([2023, 2024, 2025, 2026])
pbp = pbp.to_pandas() 
print(pbp.shape) 
print(pbp.columns.tolist())
print(pbp.head())
#Separating it into the pre snap features that can be used to predict the play type (run vs pass)
X = ['yardline_100','half_seconds_remaining','game_seconds_remaining','drive','down','goal_to_go','ydstogo',
     'posteam_timeouts_remaining','defteam_timeouts_remaining','score_differential']
y = ['play_type']

pbp_cut = pbp[X + y]
print(pbp_cut.columns.tolist())

pbp_cut = pbp_cut[pbp_cut['play_type'].isin(['run', 'pass'])] #Splitting the dataset so that it only includes runs and passes
pbp_cut = pbp_cut[pbp_cut['down'].isin([1,2,3])] #Splitting the dataset so that it only includes runs and passes
print(pbp_cut.describe())
print(pbp_cut.isnull().sum())
y = (pbp_cut['play_type'] == 'pass').astype(int) #Changing it into 1 for pass and 0 for run
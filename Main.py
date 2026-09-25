import nflreadpy as nfl
import pandas as pd
pbp = nfl.load_pbp([2023, 2024, 2025, 2026])
pbp = pbp.to_pandas() 
print(pbp.shape) 
print(pbp.columns.tolist())
print(pbp.head())
X = ['yardline_100','side_of_field','half_seconds_remaining','game_seconds_remaining','game_half','drive',
                     'down','goal_to_go','ydstogo','posteam_timeouts_remaining','defteam_timeouts_remaining',
                     'score_differential']
y = ['play_type']

pbp_cut = pbp[X + y]
print(pbp_cut.columns.tolist())
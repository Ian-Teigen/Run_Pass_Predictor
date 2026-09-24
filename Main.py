import nflreadpy as nfl
pbp = nfl.load_pbp([2023, 2024, 2025, 2026])
pbp = pbp.to_pandas() 
print(pbp.shape) 
print(pbp.columns.tolist())
#Imports
import nflreadpy as nfl
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt

#Data exploration and preparation
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

target = (pbp_cut['play_type'] == 'pass').astype(int) #Changing it into 1 for pass and 0 for run

#Model fitting
X_train, X_test, y_train, y_test = train_test_split(pbp_cut[X], target, test_size = 0.2, random_state=42)

xgb_model = XGBClassifier(
    n_estimators = 200,
    max_depth = 5,
    learning_rate = 0.1
)

xgb_model.fit(X_train, y_train)

pred_class = xgb_model.predict(X_test) #Predicts explicitly run (0) or pass (1)
pred_prob = xgb_model.predict_proba(X_test)[:, 1] #Predicts the probability of a pass

accuracy = accuracy_score(y_test, pred_class)
log_l = log_loss(y_test, pred_prob)

print(accuracy) #0.697
print(log_l) #0.566

importance = pd.Series(xgb_model.feature_importances_, index=X).sort_values(ascending=False) #feature importance
print(importance)

plt.figure(figsize = (10,6))
sns.barplot(x = importance.values, y = importance.index)
plt.title("Feature Importance")
plt.show()
#Imports
import nflreadpy as nfl
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt

#Data exploration and preparation
pbp_train = nfl.load_pbp([2023, 2024, 2025])
pbp_test_initial = nfl.load_pbp(2026)
pbp = pbp_train.to_pandas() 
pbp_test_initial = pbp_test_initial.to_pandas()
print(pbp.shape) 
print(pbp.columns.tolist())
print(pbp.head())

#Separating it into the pre snap features that can be used to predict the play type (run vs pass)
X = ['yardline_100','half_seconds_remaining','game_seconds_remaining','drive','down','goal_to_go','ydstogo',
     'posteam_timeouts_remaining','defteam_timeouts_remaining','score_differential']
y = ['play_type']

pbp_train = pbp[X + y]
pbp_test = pbp_test_initial[X + y]
print(pbp_train.columns.tolist())

pbp_train = pbp_train[pbp_train['play_type'].isin(['run', 'pass'])] #Splitting the dataset so that it only includes runs and passes
pbp_train = pbp_train[pbp_train['down'].isin([1,2,3])] #Splitting the dataset so that it only 1st 2nd and 3rd downs
pbp_test = pbp_test[pbp_test['play_type'].isin(['run', 'pass'])]
pbp_test = pbp_test[pbp_test['down'].isin([1,2,3])]

print(pbp_train.describe())
print(pbp_train.isnull().sum())

target = (pbp_train['play_type'] == 'pass').astype(int) #Changing it into 1 for pass and 0 for run
test_target = (pbp_test['play_type'] == 'pass').astype(int) 

#Model fitting
X_train = pbp_train[X]
y_train = target
X_test = pbp_test[X]
y_test = test_target

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

print(accuracy) #0.687
print(log_l) #0.572

importance = pd.Series(xgb_model.feature_importances_, index=X).sort_values(ascending=False) #feature importance
print(importance)

plt.figure(figsize = (8,6))
sns.barplot(x = importance.values, y = importance.index)
plt.title("Feature Importance")
plt.show()
clipped = X_test[X_test['ydstogo']<=25] #Clips the dataset at 25 or less yds to go
sns.lineplot(x = clipped['ydstogo'], y=pred_prob, hue = X_test['down'], palette='tab10')
plt.title("Play Type Prediction based on the Down and Yards to Go")
plt.show()
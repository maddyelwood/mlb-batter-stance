#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

batting_data = pd.read_csv('batting-data.csv')
bip_data = pd.read_csv('ball-in-play-data.csv')

data = pd.merge(
    batting_data, 
    right=bip_data, 
    on='player_id',
    )


# identify open stance based on avg_stance_angle
data['open_stance'] = (data['avg_stance_angle'] < 0)
data['open_stance'] = data['open_stance'].astype(int)

# calc totals of balls in play for each batter
data['total_ground'] = data['b_hit_ground'] + data['b_out_ground']
data['total_line_drive'] = data['b_hit_line_drive'] + data['b_out_line_drive']
data['total_popup'] = data['b_hit_popup'] + data['b_out_popup']
data['total_fly'] = data['b_hit_fly'] + data['b_out_fly']

data['total_balls_in_play'] = (data['total_ground'] 
                               + data['total_line_drive'] 
                               + data['total_popup'] 
                               + data['total_fly']
                                )

# calc ratio for each batter ball type
data['ground_ratio'] = data['total_ground'] / data['total_balls_in_play']
data['line_drive_ratio'] = data['total_line_drive'] / data['total_balls_in_play']
data['popup_ratio'] = data['total_popup'] / data['total_balls_in_play']
data['fly_ratio'] = data['total_fly'] / data['total_balls_in_play']

# calc most frequent batted ball type for each batter
data['most_freq'] = data[['ground_ratio', 'line_drive_ratio', 'popup_ratio', 'fly_ratio']].idxmax(axis=1)

# map most frequent batted ball type to numerical classes
ball_type_map = {
    'ground_ratio': 0,
    'line_drive_ratio': 1,
    'popup_ratio': 2,
    'fly_ratio': 3
    }

data['target'] = data['most_freq'].map(ball_type_map)

# select features for model training
X = data[['open_stance', 'ground_ratio', 'line_drive_ratio', 'popup_ratio', 'fly_ratio']]
y = data['target']

#print(data['target'].value_counts())

# split data train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# initialize rf model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# predict on test set
y_pred = model.predict(X_test)

# eval model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# classification report to see performance for each class
print(classification_report(y_test, y_pred))

# feature importances
importances = model.feature_importances_
features = X.columns

for feature, importance in zip(features, importances):
    print(f'{feature}: {importance:.4f}')



# linear regression to see connection between open stance and gb/fb
X = data[['open_stance']]
y = data['fly_ratio']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print('\n')
print('-----Linear Regression-----')
print(f'Fly Ratio R-squared: {r2:.4f}, RMSE: {rmse:.4f}')


X = data[['open_stance']]
y = data['ground_ratio']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f'Ground Ratio R-squared: {r2:.4f}, RMSE: {rmse:.4f}')


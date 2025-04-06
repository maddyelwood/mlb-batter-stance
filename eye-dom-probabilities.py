#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

'''
stance_data = pd.read_csv('batting-stance-2024.csv')
hitting_data = pd.read_csv('hitting-stats-2024.csv')

data = pd.merge(
    stance_data,
    hitting_data,
    left_on='id',
    right_on='player_id'
    )

print(data.head())
print(data.columns)

data.to_csv('batting-data.csv')
'''


data = pd.read_csv('batting-data.csv')

# asserting left eye dominance PROBABILITY for left/right handed batters
#data['left_eye_prob'] = data['side'].apply(lambda x: 0.5714 if x == 'L' else 0.3443)
data['left_eye_prob'] = data['throw_hand'].apply(lambda x: 0.5714 if x == 'L' else 0.3443)

# MONTE CARLO

# randomly assign ACTUAL eye dominance for testing based on probability
data['eye_dom'] = np.random.choice(
    ['L', 'R'],
    size=len(data),
    p=[data['left_eye_prob'].mean(), 1 - data['left_eye_prob'].mean()]    
    )

# create 'trailing eye' feature based on eye dominance
# if dom eye closer to back of box, i.e. right eye dom for rhh, hitter has 'trailing eye'
data['trailing_eye'] = ((data['side'] == 'R') & (data['eye_dom'] == 'R')) | \
                        ((data['side'] == 'L') & (data['eye_dom'] == 'L'))

# convert to binary 1/0 for easier analysis
data['trailing_eye'] = data['trailing_eye'].astype(int)

# identify open stance based on avg_stance_angle
data['open_stance'] = (data['avg_stance_angle'] < 0)
data['open_stance'] = data['open_stance'].astype(int)

# removing second unwanted index column
data = data.iloc[:,1:]

print(data[['throw_hand', 'side', 'eye_dom', 'trailing_eye', 'avg_stance_angle', 'open_stance']].head())
data.to_csv('batting-eye-data.csv')


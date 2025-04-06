#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

data = pd.read_csv('batting-eye-data.csv')

# group by mlb id to find switch hitters 
switch_hitters = data.groupby('id').filter(lambda x: x['side'].nunique() == 2)

# need to ensure that their dominant eye does not change via assignment since 
# each switch hitter has two rows

def randomize_eye_dom(row):
    if row['throw_hand'] == 'L':
        prob = 0.5714
        
    elif row['throw_hand'] == 'R':
        prob = 0.3443
        
    # randomly choose dominant eye based on probability
    eye_dom = np.random.choice([0,1], p=[prob, 1-prob])
    return eye_dom


def assign_dom_eye(data):
    for player in switch_hitters['id'].unique():
        hitter_rows = data[data['id'] == player]
        
        eye_dom = randomize_eye_dom(hitter_rows.iloc[0]) # same dom eye for both rows
        
        # assign dom eye to both rows of switch hitter
        data.loc[hitter_rows.index, 'eye_dom'] = eye_dom
        
        # recalculate trailing eye for both rows
        data.loc[hitter_rows.index, 'trailing_eye'] = np.where(
            (hitter_rows['side'] == 'R') & (eye_dom == 0) | (hitter_rows['side'] == 'L') & (eye_dom == 1), 1,0)

        data['trailing_eye'] = data['trailing_eye'].astype(int)
        
assign_dom_eye(data)


# compare performance metrics for swtich hitters based on eye dominance and stance
def compare_eye_side_performance(data):
    results = []
    for player in switch_hitters['id'].unique():
        hitter_rows = data[data['id'] == player]
        
        front_eye_data = hitter_rows[hitter_rows['trailing_eye'] == 0]
        trailing_eye_data = hitter_rows[hitter_rows['trailing_eye'] == 1]
        
        # calc performance for front and trailing eye per player
        front_eye_whiff = front_eye_data['whiff_percent'].mean() if not front_eye_data.empty else np.nan
        trailing_eye_whiff = trailing_eye_data['whiff_percent'].mean() if not trailing_eye_data.empty else np.nan
        
        # store results
        results.append({
            'id': player,
            'front_eye_whiff': front_eye_whiff,
            'trailing_eye_whiff': trailing_eye_whiff,
            'performance_difference': front_eye_whiff - trailing_eye_whiff 
            })
    
    return pd.DataFrame(results)

performance_comparison = compare_eye_side_performance(data)

print(performance_comparison)        

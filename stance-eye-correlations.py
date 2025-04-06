#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as smf

data = pd.read_csv('batting-eye-data.csv')

'''
# plot stance angle distribution
sns.boxplot(x=data['trailing_eye'], y=data['avg_stance_angle'])
plt.xticks(ticks=[0,1], labels=['Front Eye', 'Trailing Eye'])
plt.xlabel('Dominant Eye Position')
plt.ylabel('Stance Angle (Degrees)')
plt.title('Stance Angle vs. Dominant Eye Position')
plt.show()

# split groups
trailing_eye_open = data[data['trailing_eye'] == 1]['avg_stance_angle']
front_eye_open = data[data['trailing_eye'] == 0]['avg_stance_angle']

# run t-test to compare stance angles
t_stat, p_value = stats.ttest_ind(trailing_eye_open, front_eye_open)

print(f'T-Stat: {t_stat}, P-Value: {p_value}')



# plot stance openness vs closedness distribution
sns.boxplot(x=data['trailing_eye'], y=data['open_stance'])
plt.xticks(ticks=[0,1], labels=['Front Eye', 'Trailing Eye'])
plt.xlabel('Dominant Eye Position')
plt.ylabel('Count')
plt.title('Open vs Closed Stance by Eye Position')
plt.legend(['Closed Stance', 'Open Stance'])
plt.show()

# create contingency table
contingency_table = pd.crosstab(data['trailing_eye'], data['open_stance'])

# perform chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f'Chi-Square Stat: {chi2}, P-Value: {p}')
'''


# select performance metrics 
metrics = ['whiff_percent', 'oz_swing_percent', 'barrel_batted_rate', 'hard_hit_percent']

'''
# boxplots for each metric
for metric in metrics:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=data['open_stance'], y=data[metric])
    plt.xticks(ticks=[0,1], labels=['Closed Stance', 'Open Stance'])
    plt.xlabel('Stance Type')
    plt.ylabel(metric)
    plt.title(f'{metric} vs. Open/Closed Stance')
    plt.show()
'''

trailing_eye_data = data[data['trailing_eye'] == 1]
trailing_eye_grouped = data.groupby('open_stance')[['whiff_percent', 'oz_swing_percent', 'barrel_batted_rate', 'hard_hit_percent']].mean()

print(trailing_eye_grouped)

open_trailing = trailing_eye_data[trailing_eye_data['open_stance'] == 1]
closed_trailing = trailing_eye_data[trailing_eye_data['open_stance'] == 0]

t_stat1, p_val1 = stats.ttest_ind(open_trailing['whiff_percent'], closed_trailing['whiff_percent'], nan_policy='omit')
t_stat2, p_val2 = stats.ttest_ind(open_trailing['oz_swing_percent'], closed_trailing['oz_swing_percent'], nan_policy='omit')
t_stat3, p_val3 = stats.ttest_ind(open_trailing['barrel_batted_rate'], closed_trailing['barrel_batted_rate'], nan_policy='omit')
t_stat4, p_val4 = stats.ttest_ind(open_trailing['hard_hit_percent'], closed_trailing['hard_hit_percent'], nan_policy='omit')

print(f'Whiff%: T-stat = {t_stat1:.2f}, P-val = {p_val1:.4f}')
print(f'Out of Zone Swing%: T-stat = {t_stat2:.2f}, P-val = {p_val2:.4f}')
print(f'Barrel%: T-stat = {t_stat3:.2f}, P-val = {p_val3:.4f}')
print(f'Hard Hit%: T-stat = {t_stat4:.2f}, P-val = {p_val4:.4f}')

# test interaction models
for metric in metrics:
    formula = f'{metric} ~ trailing_eye + open_stance + trailing_eye:open_stance'
    model = smf.ols(formula, data=data).fit()
    print(f'\n=== {metric.upper()} ===')
    print(model.summary())
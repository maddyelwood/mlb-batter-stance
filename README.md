# MLB Batting Stance Analysis
This project explores newly released MLB Statcast batting stance data and applies machine learning techniques in an attempt to uncover insights about batters' performance based on their stance (open vs. closed). The goal is to test hypotheses around the impact of stance on batting outcomes, specifically focusing on eye dominance and batted ball types. Visual training and its effects on performance is a fascinating and underexplored area in sports, and this project aims to dive deeper into how eye dominance might influence batting mechanics and success.

## Hypotheses 
### 1. Eye Dominance
The first hypothesis I tested was that an open stance benefits hitters whose dominant eye is their trailing eye (i.e., the eye farther from the pitcher). To test this, I used a Monte Carlo simulation to randomize and assign eye dominance with probabilities derived from players' throwing hand and historical data from the National Library of Medicine ([source](https://pubmed.ncbi.nlm.nih.gov/15513026/)). This simulation allowed me to test the impact of eye dominance on batting outcomes with respect to their stance in the box.

Additionally, I examined differences in performance between when a switch hitter's front eye or trailing eye is dominant (depending on which side of the plate they are swinging from). The analysis focused on metrics such as whiff and barreled percentages to determine if dominant eye differences led to significant performance changes. 

### 2. Batted Ball Types Based on Stance
I also explored whether a batter's stance influenced the type of balls they hit into play (ground balls, line drives, pop-ups, or fly balls). Using random forest classifiers and linear regression, I built models to predict batted ball types based solely on stance and other related features. 

## Key Limitations
### 1. Limited Predictive Power
Despite using a variety of models, including Random Forests and Linear Regression, the results were inconclusive. For example, the prediction of batted ball types based on stance alone showed minimal correlation (R-squared values close to zero). This suggests the stance may not be the dominant factor in determining batted ball types, and other factors (such as pitch type and location) might play a more significant role.

### 2. Monte Carlo Simulation Assumptions
The Monte Carlo simulation used to randomize and assign eye dominance was based on probabilistic assumptions, which introduced a level of uncertainty. This approach may not fully reflect real-world scenarios, and further research would be required to refine these assumptions.

### 3. Sample Size Issues
The data on switch hitters was limited compared to one-sided batters, which made it challenging to draw statistically significant conclusions from these small subsets. This may have skewed results and reduced the model's ability to detect meaningful patterns. 

## Dataset
Data can be accessed from the Baseball Savant website: https://baseballsavant.mlb.com/

## Tools & Libraries
- Pandas
- Scikit-learn
- NumPy
- Matplotlib/Seaborn

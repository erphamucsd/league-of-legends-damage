# Predicting the Amount of Damage that a Player will do to Other Champions
Exploring how to feasibly predict the amount of damage a player will do.

## Problem Identification
In the game League of Legends, a 5 versus 5 Multiplayer Online Battle Arena 
(MOBA) game, one interesting aspect of the game that we can lend our attention 
to is the amount of damage that a player will do to other players over the 
course of a game. How much damage a player does to other players is not as 
clear-cut as other relationships in the game such as the relations between 
xp, cs, and gold, which are all very correlated.

Consequently, it might be interesting to investigate how to "predict the amount
of damage that a player will do to other champions".

All analyses are made on an esports dataset of 2022 pro play data with 149400 
rows of data and 123 different columns of data, with a particular focus on these
columns:

| Column Name | Description |
|-------------|-------------|
| damagetochampions | The amount of damage that a player did to other players in the game. |
| position | Indicates what role a player played during that game. |
| gamelength | The total duration of the game. |
| kills | The number of kills that a player has by the end of the game. |
| total cs | The number of creeps (minions) killed by a player by the end of a game. |
| earnedgold | The amount of gold that a player earned without including the passive income. |

In order to tackle this problem, I am building a regression model with the value
"damagetochampions" as the response variable. The method by which I am choosing
to evaluate this model is the root-mean-squared-error (RMSE), which I chose 
because it was easier to calculate for different groups as opposed to R-Squared
which is a little bit more involved mathematically. The reason why I chose RMSE
will be more apparent later on when I begin to evaluate the fairness of my 
model.

At time of prediction, we will know information about the statistics from the 
game and for each player for all of the columns mentioned above. We don't have
to worry about variables we do not yet know at time of prediction because the
value we want to predict is a final metric, and all of the features used are 
final measures as well.

## Baseline Model
### Feature Selection
The first order of business was hand picking features based off my understanding
of how it might affect how much damage a player will do. In order to select
columns, first I picked a few columns based on my understanding of what might
contribute to the damage-doing abilities of a champion, and then I tested a 
combination of features using a basic linear regression model to compare the 
impact of each feature using cross-validation. In the end, I selected these 
features for my baseline model:
-   earnedgold: This is one of the biggest indicators because gold allows players to buy items, and items enhance the amount of damage that a champion will do. It is a quantitative column and no encoding was necessary.
-   total cs: This is another big indicator because cs is a very important source of income in pro play league of legends. Because of the scarcity of kills, the most reliable way to generate income is to gain more cs, and more income means more gold to buy items which leads to more damage done. It is a quantitative column and no encoding was necessary.
-   kills: I also thought this was an important feature because a player that does a lot of damage should in theory have a lot of kills to go with it, so there's probably a positive correlation between the two values. It is a quantitative column and no encoding was necessary.
-   position: The final column that I added was position because the gold generating abilities of different positions vary a lot. Roles such as jng and sup are not meant to earn a lot of gold and typically, they do not support their teams primarily by doing damage. It is a nominal column, and it was one-hot encoded.

The corresponding RMSE for each combination are as follows:

| Feature Combinations | Average RMSE |
|-------------|-------------|
| All | 4798.454029 |
| CS/Position | 5973.101094 |
| Gold/Position | 4869.005149 |
| Kills/Position | 5693.740007 |
| Position | 6831.138548 |

### Baseline Model Selection and Performance
The model combining all of the features had the smallest RMSE, so it was the 
model that I selected for as the baseline model. On the split data, using the 
selected model, the RMSE was:
-   Training: 4798.
-   Test: 4789.

The similarity in the RMSE between the training and test data means that we can 
safely say that the model generalizes well to unseen data.

As the model is currently, I think the model does a pretty good job of 
predicting the damage done. Simply by predicting by earnedgold and position gets
us to a relatively low RMSE, and because the data has such high variance, it is 
quite difficult to get the RMSE any lower. 

<iframe src="assets/dtc_vs_gold.html" width=800 height=600 frameBorder=0></iframe>
Figure 1: The relationship between earnedgold and damagetochampions.

If the complex model performs similarly to the simple model, then there is not 
much more that can be done to dramatically improve the model. As it is, using a
simpler baseline model will still yield similar results, so I think that 
the current model is pretty difficult to make improvements on.

## Final Model
### Additional & Engineered Features
After carefully reviewing my baseline model, I decided to update it with some
more features:
-   Standardized gamelength: One of the ways that I made major updates to my final model was the inclusion of a new feature all together; standardized gamelength. I definitely think this feature helped me improve my model because the longer a game runs, the more opportunities a player has to deal damage.
-   Standardized kills: By standardizing kills, I lessened the effect that outliers had on the model. I think this was probably the least impactful to the model simply because the amount of kills that a player has doesn't deviate from the mean by much. Kills rarely happen in proplay games so the distribution of kills is very tightly packed.
-   Standardized by Group total cs: Standardizing total cs by group was another really important change to the model. The total cs varies dramatically between groups because jng and sup typically have less cs, so I thought that it would be important to scale it by each corresponding group to account for the differences in cs among groups.
-   Quadratic Scaled earnedgold: After examining the original plot I made between damagetochampions and earned gold, I noticed that there was a slight quadratic curve to the scatter plot. In order to account for this in the new model, I decided to use the square root of all the earnedgold values to help my model fit better, but because the curvature is so slight, the change was not at all impactful.

I packed all of these transformations into a ColumnTransform object, and set out
to begin picking which model algorithm to use.

### Final Selection
I began doing research on a variety of models, and decided to stick to Linear
and Tree models. These regression models were the most sensible to me because 
the relationship between the variables was almost linear, so by scaling the 
features appropriately linear models would yield the best results. I also 
decided to use ensemble models as well because it would return the optimal 
tree model for me. 

#### Hyperparameter Search
Because of my decision to use RandomForestRegressor, I had to first find the 
optimal hyperparameter, in order to compare it fairly to the other models. The
hyperparameter that I focused on the most was "max_depth" which essentially is 
the complexity of the decision tree. I only focused on this hyperparameter 
because I was worried about overfitting the data, so I needed to focus my 
efforts into correctly fitting the trees. In order to find the best "max_depth"
I manually iterated through a range of values:

| Max Depth | Average RMSE |
|-------------|-------------|
| max_depth_4 | 4901.503410 |
| max_depth_6 | 4747.683249 |
| max_depth_8 | 4678.063666 |
| max_depth_10 | 4737.271562 |
| max_depth_12 | 4932.527905 |

And after some experimentation, I found that the optimal "max_depth" was 8. I 
determined this by cross validating each model on the training set and compared
the average RMSE. Something important to note is that I conducted my 
hyperparameter search using DecisionTreeRegressor objects instead of 
RandomForestRegressor objects because of it took a long time to cross-validate
on RandomForestRegressor objects.

#### Model Algorithm
With the decision tree optimized, I could finally turn my attention to comparing
a variety of models. After comparing several models, I selected 4 and decided to
compare them. The RMSE performance for each model are as follows:

| Model Algorithm | Average RMSE |
|-------------|-------------|
| Lin Reg | 4765.019228 |
| RANSAC | 5200.738443 |
| Lasso | 4765.012166 |
| Forests | 4527.674721 |

RandomForestRegressor had the lowest average RMSE compared to the other models,
so I selected that algorithm as my final model. In order to compare all the 
models, I also employed cross-validation on the training data as my strategy to 
rank their performances. 

### RandomForestRegressor
In the end, after cross-validating twice (once for hyperparameters and once for
algorithms), I selected RandomForestRegressor(max_depth=8) as my final model.
Comparing on the same training and test sets as the baseline model the RMSE was:
-   Training: 4393.
-   Test: 4495.

| RMSE of Data | Baseline Model | Final Model |
|-------------|-------------|-------------|
| Training | 4798 | 4393 |
| Test | 4789 | 4495 |

The final model reduced RMSE by roughly 7-8% in both the training and test data,
so the final model does a pretty good job improving on the shortcomings of the 
baseline model, even though it is already very hard to improve upon. The 
reduction in error is a clear indicator of the final model's improved predicting 
powers.

<iframe src="assets/predicted_dtc_vs_gold.html" width=800 height=600 frameBorder=0></iframe>
Figure 2: The relationship between earnedgold and predicted damagetochampions.

The model, which was trained on all available data, predicted a 
similar shape to the original scatter plot, but with a much smaller spread.

## Fairness Analysis
### Fairness Evaluation
I wanted to evaluate the model's fairness based on the position class which and
I chose to use RMSE as my evaluation metric as opposed to R-Squared because when 
grouping the dataframe to calculate the error metric for each group, the formula
for R-Squared was too complicated. I also wanted to stay consistent with what I
had done throughout my project.

Comparing groups:
-   bot.
-   jng.
-   mid.
-   sup.
-   top.

Using RMSE, this is my observed RMSE for each group. While the differences are
pretty dramatic, to be sure I conducted a permutation test to evaluate the
probability of getting a spread such as this.

| position   |    RMSE |
|:-----------|--------:|
| bot        | 5495.59 |
| jng        | 3450.50 |
| mid        | 5369.34 |
| sup        | 2408.38 |
| top        | 4520.83 |

### Permutation with Chi-Squared
Because I was evaluating the fairness of five different groups instead of two, I
had to choose chi-squared as my test statistic. In order to use chi-squared, the 
expected value needs to be know, which in this case is the RMSE for the model as 
a whole. If the regressor was fair, we would expect that the RMSE for each group 
would be close to the RMSE for the whole model. In this case, the expected RMSE
was 4409. In order to investigate whether or not this regressor fairly evaluated 
based on groups, I set up a permutation test:
-   Null Hypothesis: The model is fair and its RMSE is nearly equal across all positions played. In other words, the groups do not deviate from the expected value.
-   Alternative Hypothesis: The model is unfair and it has higher RMSEs for some positions and lower RMSEs for other positions. In other words, the groups deviate heavily from the expected value.
- Test Statistic: Chi-Squared Value or sum((Expectation - Observation) ** 2 / Expectation).
- Significance Level: Rejection at the 5% or 0.05 significance level.

#### damagetochampions
<iframe src="assets/permutated_positions.html" width=800 height=600 frameBorder=0></iframe>
Figure 3: The distribution of permuted chi-squared values.

The p-value obtained for this permutation test was 0.00.

### Conclusion
At the 5% significance level, we can reject the notion that the observed RMSE 
of the position groups is close or equal to the expected RMSE for the groups.
This means that we reject the notion that the model fairly models all groups. 
My final analysis showed that the model did moderately better when evaluating
sup and jng, and this may be due to the fact that the data points are more 
tightly clustered at a first glance. In other words, they may perform more 
consistently across games compared to the other 3 roles.

However, as it stands, the model still did a pretty good job of modeling the 
damagetochampions values. In the end, using ensemble modeling allowed us to 
create a model with the best performance, without being too complex. 

Thank you for a great quarter!
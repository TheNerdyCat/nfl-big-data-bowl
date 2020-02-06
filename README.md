
# NFL Big Data Bowl
Kaggle Competition - How many yards will an NFL player gain after receiving a handoff?

## Introduction
This [competition]([https://www.kaggle.com/c/nfl-big-data-bowl-2020](https://www.kaggle.com/c/nfl-big-data-bowl-2020)) hosted by Kaggle and provided by NFL (with Next Gen Stats) was aimed at predicting the number of yards a run play will make. Different to a passing play, a run play is when the ball is handed (or tossed a short distance) to another player.

## Summary
The competition used incredibly detailed player tracking data for each play, incuding exact coordinates of every player (Offense and Defense), the movement direction, speed, acceleration, weight, height of every player... The list goes on, you can find the full data [here]([https://www.kaggle.com/c/nfl-big-data-bowl-2020/data](https://www.kaggle.com/c/nfl-big-data-bowl-2020/data)). 

We had to predict the probability of every possible yard, ranging from -100 to +100 (as plays can lose yards, for those who weren't aware). The metric used subsequently was the Continuous Ranked Probability Score (CRPS), calculated as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=C&space;=&space;\frac{1}{199N}&space;\sum_{m=1}^{N}&space;\sum_{n=-99}^{99}&space;(P(y&space;\le&space;n)&space;-H(n&space;-&space;Y_m))^2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?C&space;=&space;\frac{1}{199N}&space;\sum_{m=1}^{N}&space;\sum_{n=-99}^{99}&space;(P(y&space;\le&space;n)&space;-H(n&space;-&space;Y_m))^2" title="C = \frac{1}{199N} \sum_{m=1}^{N} \sum_{n=-99}^{99} (P(y \le n) -H(n - Y_m))^2" /></a>

where P is the predicted distribution, N is the number of plays in the test set, Y is the actual yardage and H(x) is the Heaviside step function.

The competition used a holdout test set for the public leaderboard initially, but for the final leaderboard results everyone's models were ran on the actual games as the 2019 season progressed. So we got to see each week how our model was performing, which was cool!

## Model

I scored in the top 9%, earning a bronze medal - which I was super proud of given that it was my first time competing on Kaggle! 

Due to the nature of the competition, feature engineering, training and submission were all done in one kernel - train.ipynb shared in this repository. 

In total, 31 features were passed to the model. My kernel focused predominantly on player locations and mechanics. 

Trigonometric functions were used in tandem with mechanics to calculate features such as the Force, Momentum, Kinetic Energy and Work, in the X or Y direction. Often these were aggregated across the Defense and Offense. Aggregations consisted of mean, max and standard deviations - I found the latter highly successful when used alongside any location specific features (such as standard deviation of Defense players' X coordinate. This shows how spread out or bunched up the Defense is). Gaps were also calculated based on location data, and the largest gaps were used as a feature. 

My model was a very simple Feed Forward Neural Network with 4 layers. RELU activations were used in the hidden layers, Dropout at the final layer and a softmax output activation function at the end. I experimented with adding more layers, Batch Normalisation, Gaussian Noise but never got better results than this simple model. In fact, this was my initial model, and I was expecting to use this only as a benchmark! Occam's Razor... 

I found particular success in my cross validation method. The model generalised quite well (CV 0.0127 - LB 0.0132) and I did not see the score fluctuate much as the weeks went on. A 5-Fold Grouped CV was used, and I split the data by game to help the model generalise better. Adam optimisation was used, and an impatient Early Stopping function also helped the model achieve faster and more effective results. In the end, each out-of-fold model was saved, and the final predictions were an ensemble of each of these 5 models. 

## Retrospective
Looking back, my model was essentially a descriptive Convolutional Neural Network. It was as if I could put into numbers what a person could see, which clearly worked well.

However I wish I had taken this a step further and plotted the play, feeding those images into a Convolutional Neural Network. Then, instead of summarising what's going on on the field using Triginometry, the model would be able to summarise by itself based on what it 'sees'. Incidentally, this is what the top solutions did.


## Requirements 

- Sign up to Kaggle and agree to the competition rules. 
- Tensorflow
- Keras
- Scikit-Learn
- tqdm

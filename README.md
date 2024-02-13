# NBA Game Outcome Prediction

![intro numbers](https://github.com/JaCoBaMbErT/Capstone_NBA_Predict/blob/main/Images/sports%20book%20numebrs.JPG)

## Business Understanding

This capstone project aims to predict the outcomes of NBA games. Leveraging historical game data, the project applies machine learning techniques to understand the patterns and factors that influence game results. RandomForestClassifier and ExtraTreesClassifier are the primary algorithms used, with a focus on optimizing their parameters for better performance.

## Data Understanding
The dataset used (nbaallelo.csv) comprises various features related to NBA games, including team performance metrics, game locations, and results. The data preprocessing step involves encoding categorical variables and generating new features to improve model predictions.

![data info](https://github.com/JaCoBaMbErT/Capstone_NBA_Predict/blob/main/Images/data%20values.png)


Overview

Features
elo_i, opp_elo_i: ELO ratings of the teams at the start of the game.
game_location: The location of the game (Home/Away).
avg_pts: The average points scored by the team up to the current game.
seasongame: The number of games played in the season by the team.
Target Variable
game_result: The outcome of the game (Win/Loss).

![heatmap]

Methodology

####Data Preprocessing: Encoding categorical variables and generating new features.
Feature Selection: Identifying relevant features for predicting game outcomes.
Model Training and Evaluation: Training RandomForestClassifier and ExtraTreesClassifier models, followed by accuracy assessment and parameter optimization.
Cross-validation: Performing cross-validation to evaluate the models' performance accurately.


##Installation
To run this project, you need Python 3.x and the following libraries:
pandas
scikit-learn
numpy


To execute the project, run the Jupyter Notebook (final.notebook.iypnb). Ensure the dataset nbaallelo.csv is in the correct path as specified in the notebook.


##Model Performance

Initially, the RandomForestClassifier was trained with default parameters, resulting in an accuracy score of approximately 65.7%. Recognizing the potential for improvement, we conducted parameter optimization using GridSearchCV, which allowed us to fine-tune the model parameters. The optimized RandomForestClassifier demonstrated a slight improvement in accuracy, achieving approximately 65.9%. This indicates that even minor adjustments to the model parameters can lead to better predictive performance.

The ExtraTreesClassifier was also evaluated, both with default parameters and after cross-validation. The cross-validation process, involving 5 folds, provided a more robust evaluation of the model's performance, accounting for variance in the training data. The accuracy scores across the folds were consistent, with a mean cross-validation accuracy of approximately 65.9% and a low standard deviation. This consistency underscores the model's reliability in predicting game outcomes across different data subsets.

##Parameter Optimization

Parameter optimization emerged as a crucial step in enhancing model performance. For the RandomForestClassifier, the best parameters identified included a maximum depth of 9, a max_features setting of 'log2', a minimum samples leaf of 8, a minimum samples split of 2, and 366 estimators. These optimized parameters contributed to the model's improved accuracy, highlighting the importance of fine-tuning to adapt the model to the specific characteristics of the dataset.

##Insights from Cross-validation

Cross-validation provided valuable insights into the model's stability and performance across different subsets of the data. The consistent accuracy scores across folds suggest that the ExtraTreesClassifier model is robust, with minimal overfitting to the training data. This reliability is crucial for practical applications of the model in predicting game outcomes, where the ability to generalize from historical data to future games is paramount.


![confusion](https://github.com/JaCoBaMbErT/Capstone_NBA_Predict/blob/main/Images/final%20confusion.png)

## Recomendations

The results of this project underscore the potential of machine learning models to predict sports outcomes with a reasonable degree of accuracy. While the improvements from parameter optimization were modest, they were nonetheless critical in enhancing the models' predictive power. The consistency in cross-validation results further assures us of the models' reliability and their potential utility in applications such as sports betting, team strategy development, and fan engagement initiatives.

Future work could explore additional features, alternative modeling techniques, and more advanced forms of parameter optimization to further improve predictive accuracy. Additionally, incorporating more granular data, such as player-level statistics and real-time game conditions, could offer deeper insights and potentially higher accuracy in game outcome predictions, as well setting up a rolling five game model to incorporate from the current season.




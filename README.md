# NBA Game Outcome Prediction

![intro numbers](https://github.com/JaCoBaMbErT/Capstone_NBA_Predict/blob/main/Images/sports%20book%20numebrs.JPG)

## Business Understanding

This capstone project aims to predict the outcomes of NBA games. Leveraging historical game data, the project applies machine learning techniques to understand the patterns and factors that influence game results. RandomForestClassifier and ExtraTreesClassifier are the primary algorithms used, with a focus on optimizing their parameters for better performance.

## Data Understanding
The dataset used (nbaallelo.csv) comprises various features related to NBA games, including team performance metrics, game locations, and results. The data preprocessing step involves encoding categorical variables and generating new features to improve model predictions.

![data info](https://github.com/JaCoBaMbErT/Capstone_NBA_Predict/blob/main/Images/data%20values.png)


## Overview

Features
elo_i, opp_elo_i: ELO ratings of the teams at the start of the game.
game_location: The location of the game (Home/Away).
avg_pts: The average points scored by the team up to the current game.
seasongame: The number of games played in the season by the team.
Target Variable
game_result: The outcome of the game (Win/Loss).

![heatmap](https://github.com/JaCoBaMbErT/Capstone_NBA_Predict/blob/main/Images/heatmap.png)


## Data Preprocessing: Encoding categorical variables and generating new features.
Feature Selection: Identifying relevant features for predicting game outcomes.
Model Training and Evaluation: Training RandomForestClassifier and ExtraTreesClassifier models, followed by accuracy assessment and parameter optimization.
Cross-validation: Performing cross-validation to evaluate the models' performance accurately.


## Installation
To run this project, you need Python 3.x and the following libraries:
pandas
scikit-learn
numpy


To execute the project, run the Jupyter Notebook (final.notebook.iypnb). Ensure the dataset nbaallelo.csv is in the correct path as specified in the notebook.


## Model Performance

The final model for this project was developed through a rigorous process of feature selection, model selection, and hyperparameter tuning. We employed an XGBoost classifier, a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework, which is renowned for its performance in classification tasks.

After initial exploratory data analysis and preprocessing, we used RandomizedSearchCV for an extensive search across a wide hyperparameter space. This helped us in quickly identifying promising areas for parameters that influence the model's performance. To fine-tune these parameters, we subsequently applied GridSearchCV, which performed an exhaustive search over a refined hyperparameter grid around the promising values identified in the previous step.

The model's performance was evaluated using a variety of metrics to provide a comprehensive overview of its predictive capabilities:

Accuracy: The model achieved an accuracy of 67.98%, indicating a fair level of correct predictions across the dataset.

Precision: With a precision of 68.28%, the model shows a satisfactory level of reliability in its positive predictions.

Recall: The recall of 67.87% reflects the model's ability to identify the majority of positive instances correctly.

F1 Score: An F1 Score of 68.07% suggests that the model maintains a balanced performance between precision and recall.

ROC AUC: A score of 74.61% in the ROC AUC metric indicates good discriminative ability between the positive and negative classes.

A confusion matrix was also generated to visualize the model's performance, providing a clear picture of the true positives and negatives, as well as the false positives and negatives. This matrix is crucial for understanding the model in the context of its ability to predict the winning and losing teams in a sports game scenario.

The model's performance metrics suggest it can serve as a reliable tool for predictive analysis in the domain it was designed for. While it demonstrates a solid foundation, further improvements could be made by exploring additional feature engineering, model ensembling, and domain-specific adjustments.

## Parameter Optimization

Initially, we used RandomizedSearchCV to explore a wide range of hyperparameters. This method randomly selects a subset of parameter combinations from a specified distribution. The advantage of this approach is its efficiency, allowing us to cover a broad spectrum of possibilities without the exhaustive computation required for testing every possible combination. Key parameters such as max_depth, min_child_weight, subsample, colsample_bytree, gamma, learning_rate, and n_estimators were varied. This stage helped us quickly identify promising regions within the hyperparameter space where the model showed improved performance.

With insights gained from the randomized search, we moved to a more focused and precise method—GridSearchCV. This technique conducted an exhaustive search over a specified parameter grid, but this time with narrower ranges centered around the best values found during the randomized search. This meticulous approach allowed us to fine-tune the hyperparameters, ensuring the model was optimized for the highest accuracy. Parameters were adjusted in small increments to discern the combination that yielded the best model performance according to our evaluation metrics, such as accuracy and F1 score.

The outcome of this two-step optimization process was a set of hyperparameters that were ideally suited for our XGBoost classifier, balancing the model's ability to accurately predict outcomes while avoiding overfitting. This optimized model demonstrated improved accuracy and generalization capability on unseen data, as evidenced by its performance metrics

![confusion](https://github.com/JaCoBaMbErT/Capstone_NBA_Predict/blob/main/Images/final%20confusion.png)


## Results & Recomendations

The results of this project underscore the potential of machine learning models to predict sports outcomes with a reasonable degree of accuracy. While the improvements from parameter optimization were modest, they were nonetheless critical in enhancing the models' predictive power. The consistency in cross-validation results further assures us of the models' reliability and their potential utility in applications such as sports betting, team strategy development, and fan engagement initiatives.

Future work could explore additional features, alternative modeling techniques, and more advanced forms of parameter optimization to further improve predictive accuracy. Additionally, incorporating more granular data, such as player-level statistics and real-time game conditions, could offer deeper insights and potentially higher accuracy in game outcome predictions, as well setting up a rolling five game model to incorporate from the current season.

## Repo Structure
-Data
-Images
-Notebook
-Read.me
-Game prediction.pdf



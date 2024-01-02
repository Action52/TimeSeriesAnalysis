import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import math

def gradient_boosting(train_df,test_df,features,target):
    
    """
    Creates a gradient boosting model and uses this model to generate predictions
    the train and test set
    :param train_df:
    :param test_df:
    :param features:
    :param target: 
    :return: gradient boosting model, train predictions, test predictions, 
             R2 score train, R2 score test
    """

    #drop null values (if we have moving average as target)
    features_all = features + [target]
    train_df = train_df[features_all].dropna()

    #define input and target variables
    Xtrain = train_df[features]
    Xtest = test_df[features]
    ytrain = train_df[target] 
    ytest = test_df['Close']

    #split date into day, month, year 
    if 'date' in features:
        Xtrain = split_date(Xtrain)
        Xtest = split_date(Xtest)
        Xtrain = Xtrain.drop(columns='date')
        Xtest = Xtest.drop(columns='date')

    #create gradient boosting model
    model = create_model(Xtrain,ytrain)
    
    #create predictions for train and test data
    trainPredictions, trainR2 = gradient_boosting_prediction(model,Xtrain,ytrain)
    testPredictions, testR2 = gradient_boosting_prediction(model,Xtest,ytest)

    return model, trainPredictions, testPredictions, trainR2, testR2


def create_plot(actualData, predictions):
    """
    Creates a plot of the actual data compared to the predicted data
    :param actualData:
    :param predictions:
    """


    actualData = actualData.reset_index(drop=True)
   

    plt.plot(actualData, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Bitcoin Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()



def gradient_boosting_prediction(model,data, target):
    """
    Creates predictions and and calculate the R2 score for a given model, data 
    and target values
    :param model:
    :param data:
    :param target: 
    :return: predictions for the data and R2 score for the predictions in relation
             to the target
    """
    predictions = model.predict(data)
    r2 = r2_score(target, predictions)

    return predictions, r2 





def create_model(Xtrain,ytrain):
    """
    Creates a gradient boosting model and performs hyperparameter tuning 
    with time series cross validation
    :param Xtrain:
    :param Ytrain: 
    :return: best gradient boosting model
    """

    #search grid for the hyperparameters
    param_grid = {
    'objective': ['reg:squarederror'],
    'eval_metric': ['logloss'],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'gamma':[0,0.1,0.3,0.5], 
    'max_depth': [5, 7, 8]
    # Add more hyperparameters to tune if needed
    }

    # len(Xtrain)/n_splits - gaps * n_splits - 1  
    testSize = math.floor(len(Xtrain) / 4) - 4 - 1
    print(testSize)

    #time series split for cross validation
    tss = TimeSeriesSplit(n_splits=4, test_size=testSize, gap=1)
    
    #create model and tune hyperparameters
    xgb_model = XGBRegressor()
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=tss, scoring='r2')
    grid_search.fit(Xtrain, ytrain)

    return grid_search.best_estimator_


def create_model_val(Xtrain, ytrain, Xval, yval):
    """
    Creates a gradient boosting model and performs hyperparameter tuning
    with a separate validation dataset
    :param Xtrain: Training features
    :param ytrain: Training labels
    :param Xval: Validation features
    :param yval: Validation labels
    :return: best gradient boosting model
    """

    # Search grid for the hyperparameters
    param_grid = {
        'objective': ['reg:squarederror'],
        'eval_metric': ['logloss'],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5],
        'gamma': [0, 0.1, 0.3, 0.5],
        'max_depth': [5, 7, 8]
        # Add more hyperparameters to tune if needed
    }

    # Create model and tune hyperparameters
    xgb_model = XGBRegressor()
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=4, scoring='r2')
    grid_search.fit(Xtrain, ytrain, eval_set=[(Xval, yval)], early_stopping_rounds=10, verbose=False)

    return grid_search.best_estimator_


def split_date(df):
    """
    Splits the date column of a dataframe into day, month and year column
    :param df:
    :param Ytrain: 
    :return: dataframe
    """
    df_date = df.copy()
    df_date['year']  = df_date['date'].dt.year
    df_date['month']  = df_date['date'].dt.month
    df_date['day']  = df_date['date'].dt.day

    return df_date
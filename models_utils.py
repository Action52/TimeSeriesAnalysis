import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_forecast_metrics(df, actual_col, benchmark_cols):
    """
    Calculate statistical metrics for each benchmark column against the actual column.

    :param df: DataFrame with actual and predicted data.
    :param actual_col: Column name of the actual values.
    :param benchmark_cols: List of column names for benchmark (predicted) values.
    :return: DataFrame with calculated metrics for each benchmark column.
    """
    metrics = pd.DataFrame(index=['MAE', 'MSE', 'RMSE', 'MAPE', 'R2'])

    df = df.dropna(subset=[actual_col])

    for col in benchmark_cols:
        df_copy = df.copy()
        df_copy = df_copy.dropna(subset=[col])
        actual = df_copy[actual_col]
        predicted = df_copy[col]

        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = r2_score(actual, predicted)

        metrics[col] = [mae, mse, rmse, mape, r2]

    return metrics
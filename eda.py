import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import altair as alt
import altair_viewer
import statsmodels.api as sm
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from ydata_profiling import ProfileReport

from sklearn.model_selection import train_test_split

from typing import Tuple, Any

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 20, 8


def load_train_test_data(
        file_name: str,
        currency: str,
        train_size: float = 0.85) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the original dataset, resamples the dates, and imputes the data to
    achieve a consistent time series before splitting in train and test.
    :param file_name:
    :param currency:
    :param train_size:
    :return: A tuple with the train dataframe and test dataframe.
    """

    df = load_data(file_name, currency, plot_result=False)
    df = resample_dates(df)
    df = impute_missing(df)

    train_df, test_df = split_data(df, train_size)

    return train_df, test_df


def preprocess(df: pd.DataFrame, show_results: bool = True,
               windows: tuple=(7, 30, 90, 365)):
    charts = []
    df, chart = smooth_df(df, windows, plot_result=show_results)
    charts.append(chart)
    df, chart = calculate_bollinger_bands(df, window=30)
    charts.append(chart)
    return df, charts


def load_data(file_name: str,
              currency: str,
              plot_result: bool = False) -> pd.DataFrame:
    """

    :param file_name: i.e: exchange_v2.csv
    :param currency: i.e: EURUSD=X
    :param plot_result: i.e: True
    :return: The dataframe with the selected currency data
    """
    df = pd.read_csv(file_name)
    df = df.loc[df['currency'] == currency]
    if plot_result:
        alt.Chart(df).mark_line().encode(
            x=alt.X('date:T', axis=alt.Axis(format='%Y-%m')),
            y=alt.Y('Close:Q', scale=alt.Scale(
                domain=[df['Close'].min() - 0.05,
                        df['Close'].max() + 0.05]))
        ).properties(
            width=1000,
            height=300,
            title=f"{currency} Exchange Rate"
        )
    return df


def check_nulls(df: pd.DataFrame, print_df: bool =False):
    """
    Checks if there are nulls in the columns.
    :param df: The dataframe with the financial data.
    :param print_df: To print the null count per columns.
    :return:
    """
    try:
        if print_df:
            print(df.isnull().sum())
        assert df.isnull().sum().sum() == 0
    except AssertionError as e:
        print(f"{e} There are nulls across the columns.")


def resample_dates(df: pd.DataFrame, plot_result: bool=False) -> pd.DataFrame:
    """
    Resamples the dataset to have consecutive, indexed dates.
    :param df: The dataset to resample.
    :param plot_result: Plot the result after resampling.
    :return:
    """
    currency = df.iloc[0]['currency']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True, drop=False)
    df_res = df.resample('D').asfreq()
    if plot_result:
        alt.Chart(df_res).mark_line().encode(
            x=alt.X('date:T', axis=alt.Axis(format='%Y-%m')),
            y=alt.Y('Close:Q', scale=alt.Scale(
                domain=[df_res['Close'].min() - 0.05,
                        df_res['Close'].max() + 0.05]))
        ).properties(
            width=1000,
            height=300,
            title=f"{currency} Exchange Rate including weekends"
        )
    return df_res


def impute_missing(df: pd.DataFrame, method='bfill',
                   plot_result: bool=False) -> pd.DataFrame:
    """
    Imputes the missing data rows with the immediate previous value (acceptable
    for forex data).
    :param df: DataFrame to use.
    :param method: Default 'bfill'.
    :param plot_result: Plot result after imputation.
    :return:
    """
    currency = df.iloc[0]['currency']
    df.fillna(method=method, inplace=True)

    check_nulls(df)

    if plot_result:
        alt.Chart(df).mark_line().encode(
            x=alt.X('date:T', axis=alt.Axis(format='%Y-%m')),
            y=alt.Y('Close:Q', scale=alt.Scale(
                domain=[df['Close'].min() - 0.05,
                        df['Close'].max() + 0.05]))
        ).properties(
            width=1000,
            height=300,
            title=f"{currency} Exchange Rate (Imputed)"
        )

    return df


def smooth_df(df: pd.DataFrame, windows: tuple = (7, 30, 90),
              plot_result: bool = False) -> Any:
    """
    Smooths the dataframe with simple moving averages.
    Each MA is added as a new column to the df.
    :param df: The df to smooth.
    :param windows: Window sizes to use, accepts multiple windows.
    :param plot_result: To plot the different smoothened versions of the df.
    :return:
    """
    chart = None

    for window in windows:
        df[f"mov{window}"] = df['Close'].rolling(window).mean()

    window_cols = [f"mov{window}" for window in windows]
    window_cols.append('Close')  # Include 'Close' in the fold

    # Calculate the overall min and max values across all moving averages and Close
    min_value = df[window_cols].min().min()
    max_value = df[window_cols].max().max()

    if plot_result:
        currency = df.iloc[0]['currency']

        # Create a combined chart
        chart = alt.Chart(df).transform_fold(
            fold=window_cols,
            as_=['variable', 'value']
        ).mark_line().encode(
            x=alt.X('date:T', axis=alt.Axis(format='%Y-%m')),
            y=alt.Y('value:Q', scale=alt.Scale(domain=[min_value - 0.03, max_value + 0.03])),
            color='variable:N',
            opacity=alt.condition(
                alt.datum.variable == 'Close', alt.value(0.25), alt.value(1)
            )
        ).properties(
            width=800,
            height=300,
            title=f"{currency} Exchange Rate (Smoothened with Moving Averages)"
        )

    return df, chart


def calculate_bollinger_bands(df: pd.DataFrame, window: int = 30, stds: int = 3,
                              plot_result: bool = True) -> Any:
    """
    Calculates a boundary for outliers based on bollinger bands method. Here, we
    consider an outlier any value lying beyond n standard deviations of the data.

    :param df: The df.
    :param window: Rolling window to get the band.
    :param stds: Number of standard deviations to consider.
    :param plot_result: Plot the result.
    :return:
    """
    final_chart = None
    df[f"band"] = df['Close'].rolling(window).std() * stds
    df[f"upperBand"] = (df[f'mov{window}'] +
                                df['Close'].rolling(window).std() * stds)
    df[f"lowerBand"] = df[f'mov{window}'] + df[
        'Close'].rolling(window).std() * (-1 * stds)

    # Calculate the overall min and max values across all moving averages
    min_value = df[[f'mov{window}']].min().min()
    max_value = df[[f'mov{window}']].max().max()

    if plot_result:
        currency = df.iloc[0]['currency']
        # Create the area chart for the Bollinger Bands
        band_area = alt.Chart(df).mark_area(opacity=0.3,
                                            color='lightblue').encode(
            x=alt.X('date:T', axis=alt.Axis(format='%Y-%m')),
            y=alt.Y(f'lowerBand:Q', scale=alt.Scale(
                domain=[min_value - 0.03, max_value + 0.03])),
            y2=f'upperBand:Q'
        )

        # Create the line chart for the moving average
        mov_line = alt.Chart(df).mark_line(color='orange').encode(
            x=alt.X('date:T', axis=alt.Axis(format='%Y-%m')),
            y=alt.Y('Close:Q', scale=alt.Scale(
                domain=[min_value - 0.03, max_value + 0.03])),
        )

        # Create the points chart for threshold surpassing
        threshold_points = alt.Chart(df).mark_point(color='red').encode(
            x=alt.X('date:T', axis=alt.Axis(format='%Y-%m')),
            y=alt.Y('Close:Q'),
            opacity=alt.condition(
                (alt.datum.Close > alt.datum.upperBand) | (
                            alt.datum.Close < alt.datum.lowerBand),
                alt.value(1),
                # if the condition is true, the point is fully opaque
                alt.value(0)
                # if the condition is false, the point is not shown
            )
        )

        # Combine the area, line, and points charts
        final_chart = alt.layer(band_area, mov_line,
                                threshold_points).properties(
            width=800,
            height=300,
            title=f"{currency} Bollinger Bands"
        )

    return df, final_chart


def split_data(df: pd.DataFrame, train_size: float = 0.85,
               plot_result: bool =True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data on train and test
    :param df:
    :param train_size: Proportion of train size
    :param plot_result: Plot result based on split data
    :return:
    """
    original_cols = ["date", "Open", "High", "Low", "Close", "Adj Close",
                     "Volume", "currency"]
    df = df[original_cols]

    df_train = df.iloc[0:int(df.shape[0] * train_size)]
    df_test = df.iloc[int(df.shape[0] * train_size):]

    if plot_result:
        df_train['Close'].plot(legend="Train Data")
        df_test['Close'].plot(legend="Test Data")

    return df_train, df_test



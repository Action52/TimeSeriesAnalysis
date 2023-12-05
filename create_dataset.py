import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from api import *
from tqdm import tqdm_notebook


def fetch_currency_data(pairs, start_date, end_date):
    """
    Fetch currency exchange data for given pairs between start and end dates.
    """
    data = {}
    for pair in pairs:
        data[pair] = yf.download(pair, start=start_date, end=end_date)
    return data


def create_combined_dataset(start_date, end_date, currency_pairs):
    """
    Create a combined dataset of Amadeus flight data and Yahoo Finance currency data.
    """
    currency_data = fetch_currency_data(currency_pairs, start_date, end_date)
    combined_data = []

    # Add currency data columns for each currency pair
    for pair in currency_pairs:
        pair_data = {}
        pair_data["date"] = currency_data[pair]["Close"].index

        for col in currency_data[pair].columns:
            pair_data[col] = currency_data[pair][col].values
        pair_data["currency"] = pair
        df = pd.DataFrame(pair_data)
        combined_data.append(df)
        
    # Concatenate the DataFrames in the array
    combined_df = pd.concat(combined_data, ignore_index=True)
    return combined_df


def transform_all_columns_to_rows(df):
    df_transformed_list = []
    
    for column_name in df.columns:
        df_transformed = pd.melt(df, id_vars=df.columns.difference([column_name]), value_vars=[column_name],
                                 var_name='Attribute', value_name='NewValue')
        df_transformed_list.append(df_transformed)
    
    df_combined = pd.concat(df_transformed_list, ignore_index=True)
    
    return df_combined


# Example Usage
if __name__ == "__main__":
    start_date = '2014-09-17'
    end_date = '2023-12-02'
    currency_pairs = ['EURUSD=X', 'BTC-EUR', 'BTC-USD']  # Adjust as needed

    combined_df = create_combined_dataset(start_date, end_date, currency_pairs)
    combined_df.to_csv('exchange_v2.csv', index=False)

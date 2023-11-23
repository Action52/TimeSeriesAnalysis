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


def create_combined_dataset(start_date, end_date, currency_pairs, origin,
                            destination, currency):
    """
    Create a combined dataset of Amadeus flight data and Yahoo Finance currency data.
    """
    amadeus = AmadeusAPI(os.environ["AMADEUS_ID"], os.environ["AMADEUS_SECRET"])
    flight_data = amadeus.get_flight_price_analysis_bulk(origin, destination,
                                                         start_date, end_date,
                                                         currency)

    currency_data = fetch_currency_data(currency_pairs, start_date, end_date)

    combined_data = []

    # print(flight_data)

    for date in AmadeusAPI.create_date_range(start_date, end_date):
        flight_info = next((item for item in flight_data if
                            item is not None and
                            len(item['data']) > 0 and
                            item['data'][0]['departureDate'] == date), None)

        combined_row = {'date': date}

        if flight_info:
            # Extracting data from the Amadeus API response
            flight_data_item = flight_info['data'][0]
            price_metrics = {metric['quartileRanking']: float(metric['amount'])
                             for metric in flight_data_item['priceMetrics']}
            combined_row.update({
                'origin': flight_data_item['origin']['iataCode'],
                'destination': flight_data_item['destination']['iataCode'],
                'currency_code': flight_data_item['currencyCode'],
                'minimum_price': price_metrics.get('MINIMUM'),
                'first_quartile_price': price_metrics.get('FIRST'),
                'median_price': price_metrics.get('MEDIUM'),
                'third_quartile_price': price_metrics.get('THIRD'),
                'maximum_price': price_metrics.get('MAXIMUM')
            })

        # Add currency data columns for each currency pair
        for pair in currency_pairs:
            pair_data = currency_data[pair]
            currency_info = pair_data.loc[
                pair_data.index == datetime.strptime(date, '%Y-%m-%d')]
            combined_row.update({
                f'{pair}_Close': currency_info['Close'].iloc[
                    0] if not currency_info.empty else None
                # Add other currency columns as needed
            })

        combined_data.append(combined_row)

    return pd.DataFrame(combined_data)


# Example Usage
if __name__ == "__main__":
    start_date = '2022-12-01'
    end_date = '2022-12-03'
    currency_pairs = ['EURMXN=X', 'EURTRY=X']  # Adjust as needed
    origin = 'MEX'
    destination = 'MAD'
    currency = 'EUR'

    combined_df = create_combined_dataset(start_date, end_date, currency_pairs,
                                          origin, destination, currency)
    print(combined_df)

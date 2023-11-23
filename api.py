import requests
import json
import os
from datetime import datetime, timedelta
from tqdm import tqdm_notebook
import logging


class AmadeusAPI:
    """
    A class to interact with the Amadeus Flight Price Analysis API.
    """

    def __init__(self, client_id, client_secret):
        """
        Initialize the API class with client credentials.

        :param client_id: Your Amadeus client ID.
        :param client_secret: Your Amadeus client secret.
        """
        self.base_url = "https://test.api.amadeus.com"
        self.token = None
        self.client_id = client_id
        self.client_secret = client_secret

    def authenticate(self):
        """
        Authenticate with the Amadeus API to obtain an access token.
        """
        auth_url = f"{self.base_url}/v1/security/oauth2/token"
        payload = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        response = requests.post(auth_url, data=payload)
        if response.status_code == 200:
            self.token = response.json().get('access_token')
        else:
            raise Exception("Authentication failed")

    def get_flight_price_analysis(self, origin, destination, departure_date,
                                  currency):
        """
        Get flight price analysis for a specific route and date with a specified currency.

        :param origin: IATA code for the origin airport.
        :param destination: IATA code for the destination airport.
        :param departure_date: Departure date in 'YYYY-MM-DD' format.
        :param currency: Currency code (e.g., 'USD', 'EUR').
        :return: Flight price analysis data as a JSON object.
        """
        if not self.token:
            self.authenticate()

        url = f"{self.base_url}/v1/analytics/itinerary-price-metrics"
        headers = {
            'Authorization': f'Bearer {self.token}'
        }
        params = {
            'originIataCode': origin,
            'destinationIataCode': destination,
            'departureDate': departure_date,
            'currencyCode': currency
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Failed to retrieve flight price analysis")

    def get_flight_price_analysis_bulk(self, origin, destination, start_date,
                                       end_date, currency):
        """
        Get flight price analysis for a range of dates.

        :param origin: IATA code for the origin airport.
        :param destination: IATA code for the destination airport.
        :param start_date: Start date in 'YYYY-MM-DD' format.
        :param end_date: End date in 'YYYY-MM-DD' format.
        :param currency: Currency code (e.g., 'USD', 'EUR').
        :return: List of flight price analysis data for each date.
        """
        date_list = self.create_date_range(start_date, end_date)
        results = []

        for date in tqdm_notebook(date_list):
            try:
                result = self.get_flight_price_analysis(origin, destination,
                                                        date, currency)
                results.append(result)
            except Exception as e:
                logging.info(f"Error on {date}: {e}")
                continue
        return results

    @staticmethod
    def create_date_range(start_date, end_date):
        """
        Create a list of dates between start_date and end_date.

        :param start_date: Start date in 'YYYY-MM-DD' format.
        :param end_date: End date in 'YYYY-MM-DD' format.
        :return: List of dates in 'YYYY-MM-DD' format.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        date_list = [(start + timedelta(days=x)).strftime("%Y-%m-%d") for x in
                     range((end - start).days + 1)]
        return date_list


# Example Usage
if __name__ == "__main__":
    client_id = os.environ["AMADEUS_ID"]
    client_secret = os.environ["AMADEUS_SECRET"]

    amadeus = AmadeusAPI(client_id, client_secret)
    try:
        bulk_data = amadeus.get_flight_price_analysis_bulk("MEX", "MAD",
                                                           "2022-12-01",
                                                           "2022-12-03", "USD")
        for data in bulk_data:
            print(json.dumps(data, indent=4))
    except Exception as e:
        print(f"Error: {e}")

import requests
import json
import os

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


# Example Usage
if __name__ == "__main__":
    client_id = os.environ["AMADEUS_ID"]
    client_secret = os.environ["AMADEUS_SECRET"]

    amadeus = AmadeusAPI(client_id, client_secret)
    try:
        data = amadeus.get_flight_price_analysis("MEX", "MAD", "2023-12-01",
                                                 "EUR")
        print(json.dumps(data, indent=4))
    except Exception as e:
        print(f"Error: {e}")

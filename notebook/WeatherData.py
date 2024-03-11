import pandas as pd
import requests
import os

file_csv = '~/code/harlqeuinht/which_horse/raw_data/combined_flat2_csv.csv'
df = pd.read_csv(file_csv)

### DATA CLEANING PIPELINE HERE ###

def get_race_weather(df):

    def get_co_ordinates(df):
        '''
        Given a dataframe containing a 'meeting name' column which contains race locations,
        adds two additional columns to the dataframe containing longitude and latitude
        co-ordinates for the racecourse.

        NOTES: requires API KEY to be stored in env.
        '''
        # Create a list of all unique racecourse names
        location_names = sorted(df['meeting_name'].unique())
        # Clean racecourse names so they are reconisable by the geolocation API
        locations_df = pd.DataFrame(location_names, columns=['meeting_name'])
        locations_df['location_names_cleaned'] = locations_df['meeting_name'].replace({'BANGOR-ON-DEE':'BANGOR', 'NEWMARKET (JULY)':'NEWMARKET', ' ':'_'})
        locations_df['location_names_cleaned'] = locations_df['location_names_cleaned'].str.replace(' ', '_')

        # Iterate through the locations df, generating API endpoints for each row
        for index, location in enumerate(locations_df['location_names_cleaned']):
            base_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
            api_key = os.environ['KEY']
            params = f'address={location}+racecourse&components=country:GB&key={api_key}'
            endpoint = f'{base_url}{params}'
            # Call the geolocation API, storing the results
            results = requests.get(endpoint).json()
            # Store the returned latitude and longitude data in the respective columns
            locations_df.loc[index, 'lat'] = results['results'][0]['geometry']['location']['lat']
            locations_df.loc[index, 'lng'] = results['results'][0]['geometry']['location']['lng']
        return locations_df

    def get_weather_data(df):
        '''
        Given a dataframe with the following columns: date, meeting name,
        '''
        race_dates = df['date']
        race_locations = df['meeting_name']
        unique_race_days_df = pd.DataFrame({'race_dates': race_dates, 'race_locations': race_locations})
        unique_race_days_df = unique_race_days_df.drop_duplicates()

        # Create a new dataframe containing all unique combinations of location and date
        unique_race_days_df = pd.merge(unique_race_days_df, locations_df, how='left', left_on='race_locations', right_on='meeting_name')
        # Drop the duplicated race_locations column
        unique_race_days_df.drop('race_locations', axis=1)

        # Create a function to generate API endpoints for the weather API
        def generate_endpoint(row):
            base_url = 'https://archive-api.open-meteo.com/v1/archive?'
            latitude = row['lat']
            longitude = row['lng']
            date = row['race_dates']
            params = '&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max&wind_speed_unit=mph'
            return f'{base_url}&latitude={latitude}&longitude={longitude}&start_date={date}&end_date={date}&{params}'

        # Add a column to the unique_race_days_df which contains an API endpoint for each row
        unique_race_days_df['endpoint'] = unique_race_days_df.apply(generate_endpoint, axis=1)

        # Create a function to call the weather API for each row in the unique_race_days_df
        def call_weather_api(row):
            response =requests.get(row['endpoint'])
            data = response.json()
            temp = data['daily']['temperature_2m_mean']
            precipitation = data['daily']['precipitation_sum']
            wind =data['daily']['wind_speed_10m_max']
            return temp, precipitation, wind

        # Create 3 new columns to store the weather data, and call the weather_API function to get the data
        unique_race_days_df[['temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_max']] = unique_race_days_df.apply(call_weather_api, axis=1, result_type='expand')


    #need to finish this last section so the overall function returns the data with added weather columns

import pandas as pd
import requests
import os

file_csv = '~/code/harlqeuinht/which_horse/raw_data/combined_flat2_csv.csv'
df = pd.read_csv(file_csv)

def get_weather_data(df):
    '''
    Required input is a dataframe with 'date' and 'meeting_name' columns.
    This function calls an API which obtains co-ordinates for the racecourse,
    creates a list of all unique combinations of racecourse and date, and calls
    a historic weather API to obtain weather for all of these dates.

    The function returns an 'updated df', which has 3 added columns containing
    temperature_2m_mean (degrees C), precipitation_sum (mm) and wind_speed_10m_max (mph).
    '''
    def get_co_ordinates(df):
            # Obtain list of unique racecourse names from the 'meeting_name' column
            location_names = sorted(df['meeting_name'].unique())
            # Create a locations dataframe and clean the names of racecourses to be recognisable by a geolocation API
            locations_df = pd.DataFrame(location_names, columns=['meeting_name'])
            locations_df['location_names_cleaned'] = locations_df['meeting_name'].replace({'BANGOR-ON-DEE':'BANGOR', 'NEWMARKET (JULY)':'NEWMARKET', ' ':'_'})
            locations_df['location_names_cleaned'] = locations_df['location_names_cleaned'].str.replace(' ', '_')

            # Iterate through each row of the locations dataframe, calling the geolocation API, returning co-ordinates for each
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

    # Call the function and update locations_df with the respective co-ordinates
    locations_df = get_co_ordinates(df)

    # Obtain all unique combinations of meeting name and date - returning a dataframe with all race days that took place across all venues
    def get_unique_races(df):
        unique_race_days_df = pd.DataFrame({'date': df['date'], 'meeting_name':df['meeting_name']}).drop_duplicates()
        return unique_race_days_df

    # Call the unique race days function, updating the unique race days dataframe and merging with the locations dataframe
    unique_race_days_df = get_unique_races(df)
    unique_race_days_df = pd.merge(unique_race_days_df, locations_df, how='left', left_on='meeting_name', right_on='meeting_name')

    # Generate endpoints for the weather API, accounting for date, co-ordinates and API search parameters
    def generate_endpoint(row):
            base_url = 'https://archive-api.open-meteo.com/v1/archive?'
            latitude = row['lat']
            longitude = row['lng']
            date = row['date']
            params = '&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max&wind_speed_unit=mph'
            return f'{base_url}&latitude={latitude}&longitude={longitude}&start_date={date}&end_date={date}&{params}'

    # Temporarily add a new column containing the API endpoints for workability
    unique_race_days_df['endpoint'] = unique_race_days_df.apply(generate_endpoint, axis=1)

    # Specify a function that calls the weather API on a dataframe containing an endpoint column, storing results in 3 new weather columns
    def call_weather_api(row):
            response = requests.get(row['endpoint'])
            data = response.json()

            temp = data['daily']['temperature_2m_mean'][0]
            precipitation = data['daily']['precipitation_sum'][0]
            wind = data['daily']['wind_speed_10m_max'][0]
            return temp, precipitation, wind

    # Call the API call function on the unique race days dataframe, *NOT* on the entire dataframe
    # This drastically reduces the total number of API calls as there are a significant amount of duplicates in the dataset
    unique_race_days_df[['temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_max']] = unique_race_days_df.apply(call_weather_api, axis=1, result_type='expand')

    # Merge the unique race days dataframe with the main dataframe, populating on-the-day weather for each racecourse
    updated_df = pd.merge(df, unique_race_days_df, on=['date', 'meeting_name'], how='left')
    # Drop columns created throughout this function
    updated_df = updated_df.drop(columns=['endpoint', 'lat', 'lng', 'location_names_cleaned'])
    return updated_df

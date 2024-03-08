from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import make_pipeline

from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

file_csv = '~/code/amandamor/output/combined_flat2_csv.csv'

def clean_data(df):
    df.drop(columns=['run_1_raw_post_race_rating_int', 'run_1_raw_post_race_rating_symbol', 'run_1_final_rating_int',
            'run_1_race_type', 'run_1_race_class_normalised', 'run_1_race_class', 'run_1_track_type', 'run_1_win_lose',
            'run_1_dsr', 'run_2_raw_post_race_rating_int', 'run_2_raw_post_race_rating_symbol', 'run_2_final_rating_int',
            'run_2_race_type', 'run_2_race_class', 'run_2_race_class_normalised', 'run_2_track_type', 'run_2_win_lose',
            'run_2_dsr', 'run_3_raw_post_race_rating_int', 'run_3_raw_post_race_rating_symbol', 'run_3_final_rating_int',
            'run_3_race_type', 'run_3_race_class', 'run_3_race_class_normalised', 'run_3_track_type', 'run_3_win_lose',
            'run_3_dsr', 'run_4_raw_post_race_rating_int', 'run_4_raw_post_race_rating_symbol', 'run_4_final_rating_int',
            'run_4_race_type', 'run_4_race_class', 'run_4_race_class_normalised', 'run_4_track_type', 'run_4_win_lose',
            'run_4_dsr', 'run_5_raw_post_race_rating_int', 'run_5_raw_post_race_rating_symbol', 'run_5_final_rating_int',
            'run_5_race_type', 'run_5_race_class', 'run_5_race_class_normalised', 'run_5_track_type', 'run_5_win_lose',
            'run_5_dsr', 'run_6_raw_post_race_rating_int', 'run_6_raw_post_race_rating_symbol', 'run_6_final_rating_int',
            'run_6_race_type', 'run_6_race_class', 'run_6_race_class_normalised', 'run_6_track_type', 'run_6_win_lose',
            'run_6_dsr', 'run_7_raw_post_race_rating_int', 'run_7_raw_post_race_rating_symbol', 'run_7_final_rating_int',
            'run_7_race_type', 'run_7_race_class', 'run_7_race_class_normalised', 'run_7_track_type', 'run_7_win_lose',
            'run_7_dsr', 'run_8_raw_post_race_rating_int', 'run_8_raw_post_race_rating_symbol', 'run_8_final_rating_int',
            'run_8_race_type', 'run_8_race_class', 'run_8_race_class_normalised', 'run_8_track_type', 'run_8_win_lose',
            'run_8_dsr''meeting_name', 'country_code', 'distance_unit','distance_furlongs', 'prize_money_currency',
            'jockey_allowance_unit', 'handicap_weight_unit', 'jockey_name', 'trainer_name', 'horse_name',
            'pre_race_master_rating_symbol', 'post_race_master_rating_symbol', 'post_race_master_rating_int',
            'bet365_odds', 'pmu_odds', 'meeting_id', 'distance_raw_furlongs', 'number', 'horse_id', 'age'], inplace=True)
    df['gear'] = df['gear'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['barrier'] = df[df['barrier'] <= 20]
    df['track_type'] = df['track_type'].apply(lambda x: 1 if x == 'TURF' else 0 if not pd.isna(x) else x)
    df['failed_to_finish_reason'] = df['failed_to_finish_reason'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['margin'] = df.apply(lambda row: row['distance'] if pd.isna(row['margin']) and (row['win_or_lose'] == 1 or row['failed_to_finish_reason'] == 1) else row['margin'], axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df['birth_date'] = pd.to_datetime(df['birth_date'])
    df['current_age'] = float(((df['date'] - df['birth_date']).dt.days % 365) // 30)
    df.drop(columns=['failed_to_finish_reason', 'birth_date'], inplace=True)
    return df

def categorical_features_for_onehotencoder(df):
    df['date'] = pd.to_datetime(df['date'])
    df_train = df[df['date'].dt.year != 2023]
    df_test = df[df['date'].dt.year == 2023]
    df_train.drop(columns=['jockey_id', 'tainer_id', 'margin', 'finish_position'], inplace=True)
    df_test.drop(columns=['jockey_id', 'tainer_id', 'margin', 'finish_position'], inplace=True)
    categorical_col = ['barrier', 'track_condition', 'race_type', 'track_type',
                       'race_class_normalised']
    num_col = ['distance', 'total_prize_money', 'jockey_allowance',
               'handicap_weight', 'dslr', 'official rating', 'wfa',
               'weight_adjustment', 'betfair_starting_price',
               'pre_race_master_rating_int', 'starting_price']
    ohe = ColumnTransformer(
        [
            ("onehotencoder", OneHotEncoder(), categorical_col),
        ])
    scalers = ColumnTransformer(
        [
            ("standard_scaler", StandardScaler(), num_col),
        ])
    pipeline = make_pipeline(
                        scalers,
                        ohe
                    )
    pipeline.fit(df_train)
    df_train_transformed = pipeline.transform(df_train)
    df_test_transformed = pipeline.transform(df_test)

    return df_train_transformed, df_test_transformed

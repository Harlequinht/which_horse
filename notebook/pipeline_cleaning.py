import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

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
            'run_8_dsr', 'meeting_name', 'country_code', 'distance_unit','distance_furlongs', 'prize_money_currency',
            'jockey_allowance_unit', 'handicap_weight_unit', 'jockey_name', 'trainer_name', 'horse_name',
            'pre_race_master_rating_symbol', 'post_race_master_rating_symbol', 'post_race_master_rating_int',
            'bet365_odds', 'pmu_odds', 'meeting_id', 'distance_raw_furlongs', 'number', 'horse_id', 'age', 'dam', 'sire',
            'Place', 'BSP', 'Date'], inplace=True)
    df['gear'] = df['gear'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['rating_oficial'] = df['OffR'].fillna(df['official rating'])
    df['rating_oficial'] = df['official rating'].fillna(df['OffR'])
    df = df[df['barrier'] <= 20]
    df['failed_to_finish_reason'] = df['failed_to_finish_reason'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['margin'] = df.apply(lambda row: row['distance'] if pd.isna(row['margin']) and (row['win_or_lose'] == 1 or row['failed_to_finish_reason'] == 1) else row['margin'], axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df['birth_date'] = pd.to_datetime(df['birth_date'])
    df['current_age'] = (((df['date'] - df['birth_date']).dt.days % 365) // 30).astype(float)
    df.drop(columns=['failed_to_finish_reason', 'birth_date', 'official rating', 'OffR'], inplace=True)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df


def transforming_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df.drop(columns=['jockey_id', 'tainer_id', 'margin', 'dslr','rating_oficial',
                     'last_traded_price', 'finish_position', 'event_number',
                     'post_time'], axis=1, inplace=True) # for now
    df.dropna(inplace=True) #instead of imputer for now
    df_train = df[(df['date'].dt.year != 2022) & (df['date'].dt.year != 2023)]
    df_val = df[df['date'].dt.year == 2022]
    df_test = df[df['date'].dt.year == 2023]
    df_train.drop(columns=['date'], axis=1, inplace=True)
    df_val.drop(columns=['date'], axis=1, inplace=True)
    df_test.drop(columns=['date'], axis=1, inplace=True)

    categorical_col = ['barrier', 'track_condition', 'race_type', 'track_type',
                        'race_class_normalised', 'race_class']
    num_col = ['distance', 'total_prize_money', 'jockey_allowance',
                'handicap_weight',   'wfa',
                'weight_adjustment', 'betfair_starting_price',
                'pre_race_master_rating_int', 'starting_price', 'current_age',
                'min_price', 'max_price','runners', '15_mins', '10_mins',
                '5_mins', '3_mins', '2_mins', '1_min_', ]

    categorical_preprocessor = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])
    numerical_preprocessor = Pipeline([
        ('scaler', StandardScaler())
    ])
    pipeline = ColumnTransformer([
        ('categorical', categorical_preprocessor, categorical_col),
        ('numerical', numerical_preprocessor, num_col)
    ], remainder="passthrough", sparse_threshold=0)

    pipeline.fit(df_train)
    df_train_transformed = pipeline.transform(df_train)
    df_val_transformed = pipeline.transform(df_val)
    df_test_transformed = pipeline.transform(df_test)

    categorical_feature_names = pipeline.named_transformers_['categorical'].named_steps['onehot'].get_feature_names_out(input_features=categorical_col)

    # Obter os nomes das colunas numéricas
    numerical_feature_names = num_col
    remainder_col_names = [col for col in df_train.columns if col not in (num_col+categorical_col)]

    # Combinar os nomes das colunas categóricas e numéricas
    all_feature_names = list(categorical_feature_names) + numerical_feature_names + remainder_col_names

    df_train_transformed_with_columns = pd.DataFrame(df_train_transformed, columns=all_feature_names)
    df_val_transformed_with_columns = pd.DataFrame(df_val_transformed, columns=all_feature_names)
    df_test_transformed_with_columns = pd.DataFrame(df_test_transformed, columns=all_feature_names)

    return df_train_transformed_with_columns, df_val_transformed_with_columns, df_test_transformed_with_columns


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
            'jockey_allowance_unit', 'handicap_weight_unit', 'jockey_name', 'trainer_name',
            'pre_race_master_rating_symbol', 'post_race_master_rating_symbol', 'post_race_master_rating_int',
            'bet365_odds', 'pmu_odds', #'meeting_id','horse_id',
            'distance_raw_furlongs', 'number',  'age', 'dam', 'sire',
            'betfair_starting_price', 'Date', 'id_lewagon'], inplace=True)
    df['gear'] = df['gear'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['rating_oficial'] = df['OffR'].fillna(df['official rating'])
    df['rating_oficial'] = df['official rating'].fillna(df['OffR'])
    df['finish_position'].fillna(df['Place'], inplace=True)
    df = df[df['barrier'] <= 20]
    df['failed_to_finish_reason'] = df['failed_to_finish_reason'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['margin'] = df.apply(lambda row: row['distance'] if pd.isna(row['margin']) and (row['win_or_lose'] == 1 or row['failed_to_finish_reason'] == 1) else row['margin'], axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df['birth_date'] = pd.to_datetime(df['birth_date'])
    df['current_age'] = ((df['date'] - df['birth_date']).dt.days).astype(float)
    df['Place'] = df['Place'].apply(lambda x: 99 if isinstance(x, str) and x.isalpha() else x).astype(float)
    df_sorted = df.sort_values(by=['horse_name', 'date'])
    # df_sorted['dslr'] = df_sorted['dslr'].fillna(df_sorted.groupby('horse_name')['date'].diff().dt.days)
    df_sorted.drop(columns=['failed_to_finish_reason', 'horse_name','birth_date', 'official rating', 'OffR'], inplace=True)
    df_sorted.columns = [col.lower().replace(' ', '_') for col in df_sorted.columns]

    colunas = ['15_mins', '10_mins', '5_mins', '3_mins', '2_mins', '1_min_']

    df_sem_nan = df_sorted.dropna(subset=colunas, how='all')
    number_of_nas = df_sem_nan[colunas].isna().sum().sum()
    while number_of_nas > 0:
        df_sem_nan['1_min_'] = df_sem_nan['1_min_'].fillna(df_sem_nan['2_mins'])
        df_sem_nan['2_mins'] = df_sem_nan['2_mins'].fillna(df_sem_nan['3_mins'])
        df_sem_nan['3_mins'] = df_sem_nan['3_mins'].fillna(df_sem_nan['5_mins'])
        df_sem_nan['5_mins'] = df_sem_nan['5_mins'].fillna(df_sem_nan['10_mins'])
        df_sem_nan['10_mins'] = df_sem_nan['10_mins'].fillna(df_sem_nan['15_mins'])

        df_sem_nan['15_mins'] = df_sem_nan['15_mins'].fillna(df_sem_nan['10_mins'])
        df_sem_nan['10_mins'] = df_sem_nan['10_mins'].fillna(df_sem_nan['5_mins'])
        df_sem_nan['5_mins'] = df_sem_nan['5_mins'].fillna(df_sem_nan['3_mins'])
        df_sem_nan['3_mins'] = df_sem_nan['3_mins'].fillna(df_sem_nan['2_mins'])
        df_sem_nan['2_mins'] = df_sem_nan['2_mins'].fillna(df_sem_nan['1_min_'])
        number_of_nas = df_sem_nan[colunas].isna().sum().sum()
    return df_sem_nan

def classify_group(win_or_lose, df):
    quantiles = df['win_or_lose'].quantile([0.2, 0.4, 0.6, 0.8])
    if win_or_lose <= quantiles[0.2]:
        return 5
    elif win_or_lose <= quantiles[0.4]:
        return 4
    elif win_or_lose <= quantiles[0.6]:
        return 3
    elif win_or_lose <= quantiles[0.8]:
        return 2
    else:
        return 1

def transforming_data(df, jockey_id=False, tainer_id=False):
    df['date'] = pd.to_datetime(df['date'])
    if jockey_id == True:
        df_grouped = df.groupby(by=['jockey_id']).agg({'win_or_lose': 'sum'}).sort_values(by='win_or_lose', ascending=False)
        df_grouped['jockey_class'] = df_grouped['win_or_lose'].apply(classify_group, args=(df_grouped,))
        df_grouped.drop(columns=['win_or_lose'], inplace=True)
        df = df.merge(df_grouped, how='left', left_on='jockey_id', right_on='jockey_id')
    if tainer_id == True:
        df_grouped = df.groupby(by=['tainer_id']).agg({'win_or_lose': 'sum'}).sort_values(by='win_or_lose', ascending=False)
        df_grouped['tainer_class'] = df_grouped['win_or_lose'].apply(classify_group, args=(df_grouped,))
        df_grouped.drop(columns=['win_or_lose'], inplace=True)
        df = df.merge(df_grouped, how='left', left_on='tainer_id', right_on='tainer_id')
    df.drop(columns=['jockey_id', 'tainer_id', 'margin', 'dslr','rating_oficial',
                    'last_traded_price', 'finish_position', 'event_number',
                    'pre_race_master_rating_int',
                    'post_time'], axis=1, inplace=True)# for now
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
                'weight_adjustment', 'bsp',
                 'starting_price', 'current_age',
                'min_price', 'max_price','runners', 'temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_max']

    categorical_preprocessor = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])
    numerical_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
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

    numerical_feature_names = num_col
    remainder_col_names = [col for col in df_train.columns if col not in (num_col+categorical_col)]

    all_feature_names = list(categorical_feature_names) + numerical_feature_names + remainder_col_names

    df_train_transformed_with_columns = pd.DataFrame(df_train_transformed, columns=all_feature_names)
    df_val_transformed_with_columns = pd.DataFrame(df_val_transformed, columns=all_feature_names)
    df_test_transformed_with_columns = pd.DataFrame(df_test_transformed, columns=all_feature_names)

    df_train_transformed_with_columns[numerical_feature_names] = df_train_transformed_with_columns[numerical_feature_names].astype(float)
    df_train_transformed_with_columns[categorical_feature_names] = df_train_transformed_with_columns[categorical_feature_names].astype(int)
    df_val_transformed_with_columns[numerical_feature_names] = df_val_transformed_with_columns[numerical_feature_names].astype(float)
    df_val_transformed_with_columns[categorical_feature_names] = df_val_transformed_with_columns[categorical_feature_names].astype(int)
    df_test_transformed_with_columns[numerical_feature_names] = df_test_transformed_with_columns[numerical_feature_names].astype(float)
    df_test_transformed_with_columns[categorical_feature_names] = df_test_transformed_with_columns[categorical_feature_names].astype(int)

    return df_train_transformed_with_columns, df_val_transformed_with_columns, df_test_transformed_with_columns, pipeline

from notebook.pipeline_cleaning import clean_data, transforming_data
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

file_csv = '~/code/amandamor/output/combined_flat2_csv.csv'
df_raw_data = pd.read_csv(file_csv)

df_cleaned = clean_data(df_raw_data)
df_transformed_train,  df_transformed_test = transforming_data(df_cleaned)

df_transformed_train.to_csv('./raw_data/teste_transform.csv')
models = [LinearRegression(),
          Ridge(),
          Lasso(),
          ElasticNet(),
          SGDRegressor(),
          KNeighborsRegressor(),
          SVR(kernel = "linear"),
          SVR(kernel = "poly", degree = 2),
          SVR(kernel = "poly", degree = 3),
          SVR(kernel = "rbf"),
          DecisionTreeRegressor(),
          RandomForestRegressor(),
          AdaBoostRegressor(),
          GradientBoostingRegressor()]

models_names = ["linear_regression",
                "ridge",
                "lasso",
                "elastic_net",
                "sgd_regressor",
                "kneighbors_regressor",
                "SVR_linear",
                "SVR_poly_two",
                "SVR_poly_three",
                "SVR_rbf",
                "decision_tree_regressor",
                "random_forest_regressor",
                "ada_boost_regressor",
                "gradient_boosting_regressor"]

X_train = df_transformed_train.drop(columns=['win_or_lose'])
X_test = df_transformed_test.drop(columns=['win_or_lose'])
y_train = df_transformed_train['win_or_lose']
y_test = df_transformed_test['win_or_lose']

different_test_scores = []

for model_name, model in zip(models_names, models):

    model.fit(X_train, y_train)
    different_test_scores.append(model_name, (model.score(X_test, y_test)))


comparing_regression_models = pd.DataFrame(list(zip(models_names, different_test_scores)),
                                                columns =['model_name', 'test_score'])

print(round(comparing_regression_models.sort_values(by = "test_score", ascending = False),2))

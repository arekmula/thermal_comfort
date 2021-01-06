import mlflow as mlf
import mlflow.sklearn as mlfs
import numpy as np
import pandas as pd
import pickle

from argparse import ArgumentParser
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.metrics import mean_absolute_error

from common import create_features_dataframe_from_files, add_dayofweek_to_dataframe, add_dayminute_to_dataframe,\
    drop_night_hours, drop_weekends, create_time_related_features, drop_outliers, get_month_features

mlf.set_experiment("GridSearchCV_v1_v2")
mlfs.autolog()


def main(args):
    past_samples_arr = np.arange(1, 7)
    k_best_features_arr = np.arange(1, 7)
    df_features_march, df_features_october = create_features_dataframe_from_files()

    df_features_october1 = drop_outliers(df_features_october, ("2020-10-16", "2020-10-19"))
    df_features_october1 = drop_outliers(df_features_october1, ("2020-10-31", "2020-11-02"))
    df_features_march1 = drop_outliers(df_features_march, ("2020-03-06", "2020-03-09"))

    number_of_past_samples = 3
    X_train_october, Y_train_october, X_test_october, Y_test_october = get_month_features(
        df_features_october1,
        train_upper_range="2020-10-27",
        month_name="october",
        past_samples=number_of_past_samples)
    X_train_march, Y_train_march, X_test_march, Y_test_march = get_month_features(
        df_features_march1,
        train_upper_range="2020-03-15",
        month_name="march",
        past_samples=number_of_past_samples)

    X_train = np.concatenate((X_train_october, X_train_march))
    X_test = np.concatenate((X_test_october, X_test_march))
    Y_train = np.concatenate((Y_train_october, Y_train_march))
    Y_test = np.concatenate((Y_test_october, Y_test_march))

    k_best_features = 3
    with mlf.start_run(run_name=f"k={k_best_features}, ps={number_of_past_samples}") as run:

        select_k_best = SelectKBest(f_regression, k=k_best_features)
        X_train_k = select_k_best.fit_transform(X_train, Y_train)
        X_test_k = select_k_best.transform(X_test)

        reg_name = "rf"
        if reg_name == "rf":
            reg = RandomForestRegressor(random_state=42)

        tuned_parameters = [{"n_estimators": [10, 25, 50, 75, 100, 125, 150, 175, 200],
                             "criterion": ["mae", "mse"],
                             "max_features": ["auto", "sqrt", "log2"],
                             }]

        reg = GridSearchCV(RandomForestRegressor(random_state=42),
                           param_grid=tuned_parameters,
                           scoring="neg_mean_absolute_error",
                           n_jobs=-1)
        reg.fit(X_train_k, Y_train)
        Y_predicted = reg.predict(X_test_k)

        test_mae = mean_absolute_error(Y_test, Y_predicted)

        print(f"MAE {reg_name}: {test_mae}")
        mlf.log_metric("MAE", test_mae)
        # mlfs.log_model(reg, f"{reg_name}_reg")


if __name__ == "__main__":
    parser = ArgumentParser()

    args, _ = parser.parse_known_args()

    main(args)

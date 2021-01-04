import mlflow as mlf
import mlflow.sklearn as mlfs
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.metrics import mean_absolute_error

from common import create_features_dataframe_from_files, add_dayofweek_to_dataframe, add_dayminute_to_dataframe,\
    drop_night_hours, drop_weekends, create_time_related_features, drop_outliers

mlf.set_experiment("past_samples_test1 no outliers/with night_hours and weekends")
mlfs.autolog()


def get_month_features(df_month: pd.DataFrame, train_upper_range, month_name, past_samples=5):
    # Create ground truth data by shifting measured temperature from next timestamp to previous timestamp
    df_month["temp_gt"] = df_month["temperature_middle"].shift(-1)

    # Drop NaNs from creating new columns
    df_month: pd.DataFrame = df_month.dropna()

    # Add new features
    df_month = add_dayofweek_to_dataframe(df_month)
    df_month = add_dayminute_to_dataframe(df_month)

    # TODO: Look for outliers in the data - <16.10;19.10) & <31.10;) & <6.03 18:00; 9.03)
    # TODO: Check model behaviour without some features
    # TODO: Check number of time_samples - 6 samples is the best
    # TODO: Add weather from the outside

    train_range = (df_month.index < train_upper_range)
    df_train: pd.DataFrame = df_month.loc[train_range]
    X_train, Y_train = create_time_related_features(df_train, number_of_time_samples=past_samples)
    mlf.log_param(f"{month_name}_train_size", df_train.size)

    test_range = (df_month.index > train_upper_range)
    df_test: pd.DataFrame = df_month.loc[test_range]
    X_test, Y_test = create_time_related_features(df_test, number_of_time_samples=past_samples)
    mlf.log_param(f"{month_name}_test_size", df_test.size)

    return X_train, Y_train, X_test, Y_test


def main(args):

    past_samples = np.arange(1, 7)
    df_features_march, df_features_october = create_features_dataframe_from_files()

    for past_samples in past_samples:
        with mlf.start_run(run_name=f"past_samples_{past_samples}") as run:

            df_features_october1 = drop_outliers(df_features_october, ("2020-10-16", "2020-10-19"))
            df_features_october1 = drop_outliers(df_features_october1, ("2020-10-31", "2020-11-02"))
            df_features_march1 = drop_outliers(df_features_march, ("2020-03-06", "2020-03-09"))

            X_train_october, Y_train_october, X_test_october, Y_test_october = get_month_features(
                df_features_october1,
                train_upper_range="2020-10-27",
                month_name="october",
                past_samples=past_samples)
            X_train_march, Y_train_march, X_test_march, Y_test_march = get_month_features(
                df_features_march1,
                train_upper_range="2020-03-15",
                month_name="march",
                past_samples=past_samples)

            X_train = np.concatenate((X_train_october, X_train_march))
            X_test = np.concatenate((X_test_october, X_test_march))
            Y_train = np.concatenate((Y_train_october, Y_train_march))
            Y_test = np.concatenate((Y_test_october, Y_test_march))

            reg_name="rf"
            if reg_name == "rf":
                reg = RandomForestRegressor(random_state=42)
            elif reg_name == "svr":
                reg = SVR()
            elif reg_name == "dtr":
                reg = DecisionTreeRegressor()

            reg.fit(X_train, Y_train)
            Y_predicted = reg.predict(X_test)

            test_mae = mean_absolute_error(Y_test, Y_predicted)

            print(f"MAE {reg_name}: {test_mae}")
            mlf.log_metric("MAE", test_mae)
            mlfs.log_model(reg, f"{reg_name}_reg")


if __name__ == "__main__":
    parser = ArgumentParser()

    args, _ = parser.parse_known_args()

    main(args)

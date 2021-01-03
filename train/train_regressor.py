import mlflow as mlf
import mlflow.sklearn as mlfs
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from common import create_features_dataframe_from_files, add_dayofweek_to_dataframe, add_dayminute_to_dataframe

mlf.set_experiment("Thermal comfort")
mlfs.autolog()


def main():
    with mlf.start_run(run_name="test") as run:
        mlf.set_tag("RandomForest", "RandomForest - test")
        df_features = create_features_dataframe_from_files()

        # Create ground truth data by shifting measured temperature from next timestamp to previous timestamp
        df_features["temp_gt"] = df_features["temperature_middle"].shift(-1)

        # Use last known temperature as naive method
        df_features["temp_last"] = df_features["temperature_middle"].shift(1)

        # Drop NaNs from creating new columns
        df_features: pd.DataFrame = df_features.dropna()

        df_features = add_dayofweek_to_dataframe(df_features)
        df_features = add_dayminute_to_dataframe(df_features)

        # TODO: Use information about hour and week day. Currently it's lost because index -> timestamp.
        # TODO: Use data only from timestamps between 4:00 and 16:00 and only from Monday to Friday

        train_range = (df_features.index < "2020-10-27")
        df_train: pd.DataFrame = df_features.loc[train_range]
        Y_train = df_train.pop("temp_gt")
        Y_last = df_train.pop("temp_last")  # This is made only to delete naive method from train dataframe
        X_train = df_train
        mlf.log_param("train_size", df_train.size)

        test_range = (df_features.index > "2020-10-27")
        df_test: pd.DataFrame = df_features.loc[test_range]
        Y_test = df_test.pop("temp_gt")
        Y_last = df_test.pop("temp_last")
        X_test = df_test
        mlf.log_param("test_size", df_train.size)

        print(f"MAE naive method: {mean_absolute_error(Y_test, Y_last)}")

        reg_rf = RandomForestRegressor(random_state=42)
        reg_rf.fit(X_train, Y_train)
        Y_predicted = reg_rf.predict(X_test)

        test_mae = mean_absolute_error(Y_test, Y_predicted)
        print(f"MAE RandomForest: {test_mae}")
        mlf.log_metric("MAE", test_mae)
        mlfs.log_model(reg_rf, "rf_reg")


if __name__ == "__main__":
    main()

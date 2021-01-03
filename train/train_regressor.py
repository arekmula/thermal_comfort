import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from common import create_features_dataframe_from_files


def main():
    df_features = create_features_dataframe_from_files()

    # Create ground truth data by shifting measured temperature from next timestamp to previous timestamp
    df_features["temp_gt"] = df_features["temperature_middle"].shift(-1)

    # Use last known temperature as naive method
    df_features["temp_last"] = df_features["temperature_middle"].shift(1)

    # Drop NaNs from creating new columns
    df_features: pd.DataFrame = df_features.dropna()

    # TODO: Use information about hour and week day. Currently it's lost because index -> timestamp.
    # TODO: Use data only from timestamps between 4:00 and 16:00 and only from Monday to Friday

    train_range = (df_features.index < "2020-10-27")
    df_train: pd.DataFrame = df_features.loc[train_range]
    Y_train = df_train.pop("temp_gt")
    Y_last = df_train.pop("temp_last")  # This is made only to delete naive method from train dataframe
    X_train = df_train

    test_range = (df_features.index > "2020-10-27")
    df_test: pd.DataFrame = df_features.loc[test_range]
    Y_test = df_test.pop("temp_gt")
    Y_last = df_test.pop("temp_last")
    X_test = df_test

    print(f"MAE naive method: {mean_absolute_error(Y_test, Y_last)}")

    reg_rf = RandomForestRegressor(random_state=42)
    reg_rf.fit(X_train, Y_train)
    Y_predicted = reg_rf.predict(X_test)

    print(f"MAE RandomForest: {mean_absolute_error(Y_test, Y_predicted)}")


if __name__ == "__main__":
    main()
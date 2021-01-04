import mlflow as mlf
import mlflow.sklearn as mlfs
import pandas as pd

from argparse import ArgumentParser
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.metrics import mean_absolute_error

from common import create_features_dataframe_from_files, add_dayofweek_to_dataframe, add_dayminute_to_dataframe,\
    drop_night_hours, drop_weekends

mlf.set_experiment("Thermal comfort")
mlfs.autolog()


def main(args):

    # reg_name = args.reg
    regressors = ["rf", "dtr"]
    # regressors = ["rf", "dtr", "svr", "nusvr"]

    for reg_name in regressors:
        with mlf.start_run(run_name=f"{reg_name}_test_4-16_no_weekends_alldata") as run:
            df_features = create_features_dataframe_from_files()

            # Create ground truth data by shifting measured temperature from next timestamp to previous timestamp
            df_features["temp_gt"] = df_features["temperature_middle"].shift(-1)

            # Drop NaNs from creating new columns
            df_features: pd.DataFrame = df_features.dropna()

            df_features = add_dayofweek_to_dataframe(df_features)
            df_features = add_dayminute_to_dataframe(df_features)
            df_features = drop_night_hours(df_features, lower_hour=4, upper_hour=16)
            df_features = drop_weekends(df_features)

            # TODO: Use data only from timestamps between 4:00 and 16:00 and only from Monday to Friday

            train_range = (df_features.index < "2020-10-27")
            df_train: pd.DataFrame = df_features.loc[train_range]
            Y_train = df_train.pop("temp_gt").to_numpy()
            X_train = df_train.to_numpy()
            mlf.log_param("train_size", df_train.size)

            test_range = (df_features.index > "2020-10-27")
            df_test: pd.DataFrame = df_features.loc[test_range]
            Y_test = df_test.pop("temp_gt").to_numpy()
            X_test = df_test.to_numpy()
            mlf.log_param("test_size", df_test.size)

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
    # parser.add_argument("--reg", metavar="reg", type=str, required=True)

    args, _ = parser.parse_known_args()

    main(args)

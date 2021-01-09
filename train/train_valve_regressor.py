import mlflow as mlf
import mlflow.sklearn as mlfs
import numpy as np
import pickle

from argparse import ArgumentParser
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error

from common import create_features_dataframe_from_files, add_dayofweek_to_dataframe, add_dayminute_to_dataframe,\
    drop_night_hours, drop_weekends, create_time_related_features, drop_outliers, get_month_features

mlf.set_experiment("valve_regressor_october only(to 31.10)")
mlfs.autolog()


def main(args):
    past_samples_arr = np.arange(1, 200)
    k_best_features_arr = np.arange(5, 7)
    df_features_march1, df_features_october1 = create_features_dataframe_from_files()

    # df_features_october1 = drop_outliers(df_features_october, ("2020-10-16", "2020-10-19"))
    # df_features_october1 = drop_outliers(df_features_october, ("2020-10-31", "2020-11-03"))
    # df_features_march1 = drop_outliers(df_features_march, ("2020-03-06", "2020-03-09"))

    # number_of_past_samples = 3
    for number_of_past_samples in past_samples_arr:
        X_train_october, Y_train_october, X_test_october, Y_test_october = get_month_features(
            df_features_october1,
            train_upper_range="2020-10-27",
            past_samples=number_of_past_samples,
            gt_column_name="radiator_1_valve_level")
        # X_train_march, Y_train_march, X_test_march, Y_test_march = get_month_features(
        #     df_features_march1,
        #     train_upper_range="2020-03-15",
        #     past_samples=number_of_past_samples,
        #     gt_column_name="radiator_1_valve_level")

        # X_train = np.concatenate((X_train_october, X_train_march))
        # X_test = np.concatenate((X_test_october, X_test_march))
        # Y_train = np.concatenate((Y_train_october, Y_train_march))
        # Y_test = np.concatenate((Y_test_october, Y_test_march))
        X_train = X_train_october
        X_test = X_test_october
        Y_train = Y_train_october
        Y_test = Y_test_october

        # k_best_features = 3
        for k_best_features in k_best_features_arr:
            with mlf.start_run(run_name=f"k={k_best_features}, ps={number_of_past_samples}") as run:

                select_k_best = SelectKBest(f_regression, k=k_best_features)
                X_train_k = select_k_best.fit_transform(X_train, Y_train)
                X_test_k = select_k_best.transform(X_test)

                reg_name = "rf"
                if reg_name == "rf":
                    reg = RandomForestRegressor(random_state=42)

                reg.fit(X_train_k, Y_train)
                Y_predicted = reg.predict(X_test_k)

                test_mae = mean_absolute_error(Y_test, Y_predicted)

                print(f"MAE {reg_name}: {test_mae}")
                mlf.log_metric("MAE", test_mae)

                # pickle.dump(reg, open("../models/valve_regressor.p", "wb"))
                # pickle.dump(select_k_best, open("../models/valve_feature_selector.p", "wb"))

                # mlfs.log_model(reg, f"{reg_name}_reg")


if __name__ == "__main__":
    parser = ArgumentParser()

    args, _ = parser.parse_known_args()

    main(args)

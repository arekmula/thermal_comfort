import pandas as pd
import pickle

from pathlib import Path
from typing import Tuple

from common.utils import get_devices_serial_numbers, process_supply_points_file,\
    add_dayminute_to_dataframe, add_dayofweek_to_dataframe, create_time_related_features


def perform_processing(
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str
) -> Tuple[float, float]:

    serial_numbers = get_devices_serial_numbers(path="data/additional_info.json")

    df_temperature = process_supply_points_file(temperature, serial_numbers, label="right")
    df_target_temperature = process_supply_points_file(target_temperature, serial_numbers,
                                                       device_name="radiator_1_target",
                                                       label="right")
    df_valve_level = process_supply_points_file(valve_level, serial_numbers, "radiator_1_valve_level",
                                                label="right")

    df_features = pd.concat([df_temperature, df_target_temperature, df_valve_level], axis=1)
    df_features = df_features.dropna()
    df_features = df_features.sort_index()

    df_features = add_dayofweek_to_dataframe(df_features)
    df_features = add_dayminute_to_dataframe(df_features)

    X_related_features, _ = create_time_related_features(df_features, number_of_time_samples=3, create_gt=False)

    with Path('models/feature_selector.p').open("rb") as feature_selector_file:
        feature_selector = pickle.load(feature_selector_file)

    X_related_features_selected = feature_selector.transform(X_related_features)

    with Path("models/regressor.p").open("rb") as regressor_file:
        regressor = pickle.load(regressor_file)

    Y_predict = regressor.predict([X_related_features_selected[-1]])
    return Y_predict[0], 0.0

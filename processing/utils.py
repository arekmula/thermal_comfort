import pandas as pd
import pickle

from pathlib import Path
from typing import Tuple, Dict

from common.utils import get_devices_serial_numbers, process_supply_points_file, \
    add_dayminute_to_dataframe, add_dayofweek_to_dataframe, create_time_related_features


def load_models():
    with Path('models/temp_feature_selector.p').open("rb") as feature_selector_file:
        temp_feature_selector = pickle.load(feature_selector_file)
    with Path('models/valve_feature_selector.p').open("rb") as feature_selector_file:
        valve_feature_selector = pickle.load(feature_selector_file)
    with Path("models/temp_regressor.p").open("rb") as regressor_file:
        temp_regressor = pickle.load(regressor_file)
    with Path("models/valve_regressor.p").open("rb") as regressor_file:
        valve_regressor = pickle.load(regressor_file)

    models = {"temp_feature_selector": temp_feature_selector,
              "valve_feature_selector": valve_feature_selector,
              "temp_regressor": temp_regressor,
              "valve_regressor": valve_regressor}

    return models


def perform_processing(
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str,
        models: Dict
) -> Tuple[float, float]:
    serial_numbers = get_devices_serial_numbers(path="data/additional_info.json")

    df_temperature = process_supply_points_file(temperature, serial_numbers, label="right")
    df_target_temperature = process_supply_points_file(target_temperature, serial_numbers,
                                                       device_name="radiator_1_target",
                                                       label="right")
    df_valve_level = process_supply_points_file(valve_level, serial_numbers, "radiator_1_valve_level", label="right")

    df_features = pd.concat([df_temperature, df_target_temperature, df_valve_level], axis=1)
    df_features = df_features.dropna()
    df_features = df_features.sort_index()

    # df_features = add_dayofweek_to_dataframe(df_features)
    # df_features = add_dayminute_to_dataframe(df_features)

    X_temp_time_related_features, _ = create_time_related_features(df_features,
                                                                   number_of_time_samples=3,
                                                                   create_gt=False)
    X_valve_time_related_features, _ = create_time_related_features(df_features,
                                                                    number_of_time_samples=169,
                                                                    create_gt=False)

    temp_feature_selector = models["temp_feature_selector"]
    valve_feature_selector = models["valve_feature_selector"]

    X_temp_timerelated_features_selected = temp_feature_selector.transform(X_temp_time_related_features)
    X_valve_timerelated_features_selected = valve_feature_selector.transform(X_valve_time_related_features)

    temp_regressor = models["temp_regressor"]
    valve_regressor = models["valve_regressor"]

    predicted_temp = temp_regressor.predict([X_temp_timerelated_features_selected[-1]])
    predicted_valve = valve_regressor.predict([X_valve_timerelated_features_selected[-1]])

    return predicted_temp[0], predicted_valve[0]

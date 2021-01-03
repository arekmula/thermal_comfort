import json

import pandas as pd


def get_devices_serial_numbers(path: str):
    """
    Gets devices serial number and corresponding device names from given file

    :param path: path to additional_info file
    :return: dictionary holding serialNumber as a key and device name as value
    """
    try:
        with open(path) as file:
            additional_data = json.load(file)
    except FileNotFoundError as e:
        print(e)
        return None

    devices = additional_data["offices"]["office_1"]["devices"]

    devices_serial_numbers = []
    for device in devices:
        device_serial_number = {device["serialNumber"]: device["description"]}
        devices_serial_numbers.append(device_serial_number)

    return devices_serial_numbers


def process_supply_points_file(path: str, serial_numbers, device_name=None):
    """
    Creates dataframe from input supply point file. If device_name is not given it creates dataframe based on devices
    serial numbers. Otherwise it create dataframe for only one device (assume that input file has only one serialNumber).

    :param path: path to csv file containing input data
    :param serial_numbers: serial numbers of devices to slice
    :param device_name: device name if input data contains only one serial number
    :return dataframe based on input file:
    """
    try:
        df_temporary = pd.read_csv(path)
    except FileNotFoundError as e:
        print(e)
        return None

    df_temporary.rename(columns={"Unnamed: 0": "time"}, inplace=True)
    df_temporary["time"] = pd.to_datetime(df_temporary["time"])
    df_temporary.drop(columns=["unit"], inplace=True)

    # List holding dataframes for each sensor device
    df_devices = []
    if device_name is None:
        for device in serial_numbers:
            device_serial_number = list(device.keys())[0]
            device_name = list(device.values())[0]

            # Create separate dataframe for each device
            df_device = df_temporary[df_temporary["serialNumber"] == device_serial_number]
            df_device_renamed = df_device.rename(columns={"value": device_name})
            df_device_renamed.drop(columns=["serialNumber"], inplace=True)
            # Set time as index
            df_device_renamed.set_index("time", inplace=True)
            df_devices.append(df_device_renamed)
    else:
        df_device_renamed = df_temporary.rename(columns={"value": device_name})
        # Set time as index
        df_device_renamed.set_index("time", inplace=True)
        df_devices.append(df_device_renamed)

    # Create one dataframe for all devices
    df_temperatures = pd.concat(df_devices)
    # Resample device
    df_temperatures = df_temperatures.resample(pd.Timedelta(minutes=15)).mean().fillna(method="ffill")

    return df_temperatures


def create_features_dataframe_from_files() -> pd.DataFrame:
    """
    Creates one dataframe from sepearate files holding measured temperature for different devices, target temperature
    for radiator and valve level of radiator


    :return: pandas dataframe holding above values
    """
    serial_numbers = get_devices_serial_numbers(path='../data/additional_info.json')

    df_temperature = process_supply_points_file(
        '../data/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv',
        serial_numbers)

    df_target_temperature = process_supply_points_file(
        "../data/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv",
        serial_numbers,
        device_name="radiator_1_target")

    df_valve_level = process_supply_points_file(
        "../data/office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv",
        serial_numbers,
        device_name="radiator_1_valve_level"
    )

    df_features = pd.concat([df_temperature, df_target_temperature, df_valve_level], axis=1)
    df_features = df_features.dropna()

    return df_features


def add_dayofweek_to_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Adds day of week to dataframe based on dataframe index

    :param dataframe: dataframe to fill
    :return: filled dataframe
    """
    dataframe["day_of_week"] = dataframe.index.dayofweek

    return dataframe


def add_dayminute_to_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Adds minute of day to dataframe based on dataframe index

    :param dataframe: dataframe to fill
    :return: filled dataframe
    """
    dataframe["day_minute"] = dataframe.index.hour * 60 + dataframe.index.minute

    return dataframe

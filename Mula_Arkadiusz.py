import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

from processing.utils import perform_processing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    input_file = Path(args.input_file)
    results_file = Path(args.results_file)

    with open(input_file) as f:
        arguments = json.load(f)

    start = pd.Timestamp(arguments['start']).tz_localize('UTC')
    stop = pd.Timestamp(arguments['stop']).tz_localize('UTC')

    df_temperature = pd.read_csv(arguments['file_temperature'], index_col=0, parse_dates=True)
    df_target_temperature = pd.read_csv(arguments['file_target_temperature'], index_col=0, parse_dates=True)
    df_valve = pd.read_csv(arguments['file_valve_level'], index_col=0, parse_dates=True)

    df_temperature_serial_number = df_temperature[df_temperature["serialNumber"] == arguments["serial_number"]]
    # Adding label="right" means if we are resampling every 15 minutes, and the last data is from timestamps
    # 22:45, 22:47, 22:53, 22:58, it will resample this timestamps to one -> 23:00.
    # If label="left" that means that those timestamps would be missed and the last timestamp would be 22:45
    df_temperature_resampled = df_temperature_serial_number.resample(pd.Timedelta(minutes=15),
                                                                     label="right").mean().fillna(method='ffill')
    df_temperature_resampled = df_temperature_resampled.loc[start:stop]
    df_temperature_resampled['predicted'] = 0.0

    current = start - pd.DateOffset(minutes=15)
    while current < stop:
        predicted_temperature = perform_processing(
            df_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_target_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_valve.loc[(current - pd.DateOffset(days=7)):current],
            arguments['serial_number']
        )
        current = current + pd.DateOffset(minutes=15)

        df_temperature_resampled.at[current, 'predicted'] = predicted_temperature

    df_temperature_resampled.to_csv(results_file)
    print(mean_absolute_error(df_temperature_resampled["value"], df_temperature_resampled["predicted"]))


if __name__ == '__main__':
    main()

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
    df_temperature_sn = df_temperature[df_temperature['serialNumber'] == arguments['serial_number']]
    df_target_temperature = pd.read_csv(arguments['file_target_temperature'], index_col=0, parse_dates=True)
    df_valve = pd.read_csv(arguments['file_valve_level'], index_col=0, parse_dates=True)

    df_combined = pd.concat([
        df_temperature_sn.rename(columns={'value': 'temperature'}),
        df_target_temperature.rename(columns={'value': 'target_temperature'}),
        df_valve.rename(columns={'value': 'valve_level'})
    ])

    df_combined_resampled = df_combined.resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')
    df_combined_resampled = df_combined_resampled.loc[start:stop]
    df_combined_resampled['predicted_temperature'] = 0.0
    df_combined_resampled['predicted_valve_level'] = 0.0

    current = start - pd.DateOffset(minutes=15)

    while current < stop:
        predicted_temperature, predicted_valve_level = perform_processing(
            df_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_target_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_valve.loc[(current - pd.DateOffset(days=7)):current],
            arguments['serial_number']
        )
        current = current + pd.DateOffset(minutes=15)

        df_combined_resampled.at[current, 'predicted_temperature'] = predicted_temperature
        df_combined_resampled.at[current, 'predicted_valve_level'] = predicted_valve_level

    print(mean_absolute_error(df_combined_resampled["temperature"], df_combined_resampled["predicted_temperature"]))
    df_combined_resampled.to_csv(results_file)


if __name__ == '__main__':
    main()

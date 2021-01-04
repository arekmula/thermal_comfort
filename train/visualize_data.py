import pandas as pd

from matplotlib import pyplot as plt

from common import create_features_dataframe_from_files


def plot_dataframe(dataframe: pd.DataFrame, columns_to_plot):

    dataframe.plot(y=columns_to_plot)
    ax = dataframe['radiator_1_valve_level'].plot(style="--", secondary_y=True, )


def main():
    df_features_march, df_features_october = create_features_dataframe_from_files()

    columns_to_plot = []
    columns_to_plot.append("radiator_1")
    # columns_to_plot.append("temperature_wall")
    # columns_to_plot.append("temperature_window")
    columns_to_plot.append("temperature_middle")
    columns_to_plot.append("radiator_1_target")

    plot_dataframe(df_features_march, columns_to_plot)
    plot_dataframe(df_features_october, columns_to_plot)


    plt.show()


if __name__ == "__main__":
    main()

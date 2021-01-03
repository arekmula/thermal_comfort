from matplotlib import pyplot as plt

from common import create_features_dataframe_from_files


def main():
    df_features = create_features_dataframe_from_files()
    df_features.pop("radiator_1_valve_level")
    df_features.plot()
    plt.show()


if __name__ == "__main__":
    main()

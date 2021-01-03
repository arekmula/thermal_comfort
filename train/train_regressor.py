from common import create_features_dataframe_from_files


def main():
    df_features = create_features_dataframe_from_files()
    print(df_features)
    ...


if __name__ == "__main__":
    main()

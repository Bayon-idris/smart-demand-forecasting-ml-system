import pandas as pd
from collections import defaultdict

import pandas as pd

from utils import constant


def load_group_best_sales(file_path, chunksize=1_000_000):
    store_sales = defaultdict(int)
    store_products = defaultdict(set)

    reader = pd.read_csv(file_path, chunksize=chunksize)

    for chunk in reader:
        for store, group in chunk.groupby("store_id"):
            store_sales[store] += group["sales"].sum()
            store_products[store].update(group["item_number"].unique())

    summary = []

    for store in store_sales:
        summary.append(
            {
                "store_id": store,
                "total_sales": store_sales[store],
                "unique_products": len(store_products[store]),
            }
        )

    df_summary = pd.DataFrame(summary).sort_values("total_sales", ascending=False)

    return df_summary


def create_new_csv_file_based_on_store_best_sales(
    store_to_extract_id, output_file_path, file_path, features
):

    reader = pd.read_csv(file_path, chunksize=1_000_000)
    first_write = True

    with open(output_file_path, "w", newline="") as f_out:

        for chunk in reader:

            chunk_store = chunk[chunk["store_id"] == store_to_extract_id]

            if chunk_store.empty:
                continue

            chunk_store = create_time_series_features(chunk_store)

            chunk_store = add_peak_flag(chunk_store)

            chunk_store = convert_date_in_date_format(chunk_store)

            chunk_store = chunk_store.dropna()

            chunk_store = keep_only_features(chunk_store, features)

            if first_write:
                chunk_store.to_csv(f_out, index=False)
                first_write = False
            else:
                chunk_store.to_csv(f_out, index=False, header=False)

    print("Extraction terminée.")


def create_time_series_features(df):

    # LAGS
    df["lag_14"] = df.groupby("item_number")["sales"].shift(14)
    df["lag_28"] = df.groupby("item_number")["sales"].shift(28)

    # ROLLING MEAN
    df["rolling_mean_7"] = (
        df.groupby("item_number")["sales"].shift(1).rolling(window=7).mean()
    )

    df["rolling_mean_30"] = (
        df.groupby("item_number")["sales"].shift(1).rolling(window=30).mean()
    )

    # ROLLING STD
    df["rolling_std_7"] = (
        df.groupby("item_number")["sales"].shift(1).rolling(window=7).std()
    )

    return df


def add_peak_flag(df):

    df["is_peak"] = (
        df["sales"] > (df["rolling_mean_30"] + 2 * df["rolling_std_7"])
    ).astype(int)

    return df

def keep_only_features(df, features):
    existing_features = [col for col in features if col in df.columns]
    return df[existing_features]

def convert_date_in_date_format(df):
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    return df

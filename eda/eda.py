import pandas as pd
from collections import defaultdict

import pandas as pd

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
        summary.append({
            "store_id": store,
            "total_sales": store_sales[store],
            "unique_products": len(store_products[store])
        })

    df_summary = pd.DataFrame(summary).sort_values("total_sales", ascending=False)

    return df_summary

def create_new_csv_file_based_on_store_best_sales(store_to_extract_id, output_file_path, file_path):
    reader = pd.read_csv(file_path, chunksize=1_000_000)

    with open(output_file_path, "w", newline="") as f_out:
        for i, chunk in enumerate(reader):
            chunk_store = chunk[chunk["store_id"] == store_to_extract_id]
            if i == 0:
                chunk_store.to_csv(f_out, index=False)
            else:
                chunk_store.to_csv(f_out, index=False, header=False)

    print("Extraction terminée.")
    
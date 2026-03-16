model_base_path = "models/xgboost_model.pkl"
store_base_path = "data/store_sales.csv"


features = [
    "sales",
    "date",
    "wm_yr_wk",
    "wday",
    "snap",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_std_7",
    "item_subcategory",
    "item_number",
    "sell_price",
    "price_flag",
    "snap_weekend",
    "event_count",
    "event_impact"
]

target = "sales"

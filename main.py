import pandas as pd
from collections import defaultdict
from eda.eda import create_new_csv_file_based_on_store_best_sales, create_time_series_features, load_group_best_sales
from predictions.prediction import predict, predict
from utils import constant
from utils.verify_path import verify_data_path


file_path = "data/data.csv"
output_file_path = "data/store_sales.csv"
def main():
    
    verify_data_path(file_path)
    reader = pd.read_csv(file_path)
    
    # thoses  function are called once time to get the best sales per store and create a new csv file with the best sales of the store CA_3 in this case
    # best_sales_df = load_group_best_sales(file_path) 
    create_new_csv_file_based_on_store_best_sales(store_to_extract_id="CA_3", output_file_path=output_file_path, file_path=file_path ,features=constant.features)

    # verify_data_path(output_file_path)
    # df = pd.read_csv(output_file_path)
    # predict(output_file_path)

    

if __name__ == "__main__":
    main()
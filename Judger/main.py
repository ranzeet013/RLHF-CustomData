import pandas as pd
from judger import Judger

def convert_and_load_csv(df, csv_path, n=3):
    """
    Takes a DataFrame, selects the top `n` records, converts them into a CSV file, 
    saves it, and then loads it back.

    :param df: Pandas DataFrame loaded from JSON.
    :param csv_path: Path to save the CSV file.
    :param n: Number of records to select (default is 3).
    """
    try:
        print("First 2 records from JSON data:")
        print(df.head(2).to_json(orient="records", indent=4))

        top_n = df.head(n)

        top_n.to_csv(csv_path, index=False)
        print(f"\nCSV file saved successfully at: {csv_path}")
        loaded_df = pd.read_csv(csv_path)
        
        print("\nLoaded CSV Data:")
        print(loaded_df)

        return loaded_df

    except Exception as e:
        print(f"Error: {e}")

# Example usage
json_file_path = "querydata.json"
csv_file_path = "querydata.csv"

# Load JSON once
df = pd.read_json(json_file_path)
loaded_data = convert_and_load_csv(df, csv_file_path)
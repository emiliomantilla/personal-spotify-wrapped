import os
import json
import pandas as pd
import zipfile
from io import BytesIO

def load_streaming_history(data_folder="../data"):
    """
    Legacy function: Reads all JSON files in the specified data folder,
    merges them into a single DataFrame, and returns it.
    This function is maintained for backward compatibility and local testing,
    but is not used in the deployed app.
    """
    all_dfs = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(data_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.json_normalize(data)
            all_dfs.append(df)
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def ingest_spotify_zip(zip_file_bytes: bytes) -> pd.DataFrame:
    """
    Ingests a ZIP file (provided as bytes) containing one or more
    Spotify Streaming History JSON files and returns a combined DataFrame.
    This is the function used in the deployed app.
    """
    with zipfile.ZipFile(BytesIO(zip_file_bytes), 'r') as z:
        all_dfs = []
        for filename in z.namelist():
            if filename.lower().endswith('.json'):
                with z.open(filename) as f:
                    data = json.load(f)
                df = pd.json_normalize(data)
                all_dfs.append(df)
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

def main():
    """
    Main function for testing the ingest_spotify_zip function.
    
    Instead of using the legacy load_streaming_history function,
    this test automatically finds a ZIP file in the local data folder.
    
    It assumes there is exactly one ZIP file in the data folder that contains
    the extended streaming history. If multiple ZIP files are found, it uses the
    first one and prints a warning. If no ZIP file is found, it prints an error.
    """
    data_folder = os.path.join("..", "data")
    # List all ZIP files in the data folder
    zip_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.zip')]
    
    if not zip_files:
        print(f"No ZIP file found in {data_folder}. Please add a ZIP file for ingestion testing.")
        return
    
    if len(zip_files) > 1:
        print(f"Warning: Multiple ZIP files found. Using the first one: {zip_files[0]}")
    
    test_zip_path = os.path.join(data_folder, zip_files[0])
    print(f"Using ZIP file: {test_zip_path}")
    
    # Open and read the selected ZIP file in binary mode
    with open(test_zip_path, "rb") as f:
        zip_bytes = f.read()

    # Use the ingest_spotify_zip function to load data from the ZIP file
    df_history = ingest_spotify_zip(zip_bytes)
    print("Loaded streaming history from ZIP:", df_history.shape)
    print(df_history.head())

if __name__ == "__main__":
    main()

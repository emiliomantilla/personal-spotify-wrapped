# src/data_cleaning.py

import os
import pandas as pd

def clean_spotify_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and transforms the streaming history DataFrame.
    - Renames columns.
    - Converts timestamps.
    - Adds new columns (e.g., hour, day_of_week, minutes_played).
    - Drops unnecessary columns.
    """
    print("Original columns:", df.columns.tolist())
    
    # Rename columns for simplicity
    rename_dict = {
        'master_metadata_track_name': 'track_name',
        'master_metadata_album_artist_name': 'artist_name',
        'master_metadata_album_album_name': 'album_name',
        'ms_played': 'ms_played'
    }
    df.rename(columns=rename_dict, inplace=True)
    
    # Convert timestamps and add time-based columns
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
        df['hour'] = df['ts'].dt.hour
        df['day_of_week'] = df['ts'].dt.day_name()
    elif 'endTime' in df.columns:
        df['ts'] = pd.to_datetime(df['endTime'], errors='coerce')
        df['hour'] = df['ts'].dt.hour
        df['day_of_week'] = df['ts'].dt.day_name()

    # Create a new column for minutes played (if ms_played exists)
    if 'ms_played' in df.columns:
        df['minutes_played'] = df['ms_played'] / 60000.0

    # Drop unnecessary columns if they exist
    columns_to_drop = [
        "episode_name",
        "episode_show_name",
        "spotify_episode_uri",
        "audiobook_title",
        "audiobook_uri",
        "audiobook_chapter_uri",
        "audiobook_chapter_title"
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")
    
    print("Cleaned columns:", df.columns.tolist())
    return df

def clean_data(input_filepath, output_filepath):
    """
    Legacy function:
    Reads a CSV from input_filepath, cleans the data, and writes it to output_filepath.
    This function is maintained for backward compatibility.
    """
    df = pd.read_csv(input_filepath)
    df_clean = clean_spotify_df(df)
    df_clean.to_csv(output_filepath, index=False)
    print(f"Cleaned data saved to {output_filepath}")

if __name__ == "__main__":
    """
    Main function for testing the cleaning function using the output of the ingestion function.
    
    Instead of reading from a CSV file, this test:
      1. Searches the local data folder for any ZIP file (assumes there's only one ZIP file).
      2. Uses the ingest_spotify_zip function (from data_ingestion.py) to load the raw data.
      3. Cleans the data with clean_spotify_df.
      4. Prints a preview (first five rows) and the info() output to display the new structure.
    """
    # Import the ingest_spotify_zip function from data_ingestion
    from data_ingestion import ingest_spotify_zip

    # Define the path to your data folder
    data_folder = os.path.join("..", "data")
    
    # Find ZIP files in the data folder (flexible to any ZIP file name)
    zip_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.zip')]
    
    if not zip_files:
        print(f"No ZIP file found in {data_folder}. Please add a ZIP file for testing.")
    else:
        if len(zip_files) > 1:
            print(f"Warning: Multiple ZIP files found. Using the first one: {zip_files[0]}")
        test_zip_path = os.path.join(data_folder, zip_files[0])
        print(f"Using ZIP file: {test_zip_path}")

        # Open and read the ZIP file in binary mode
        with open(test_zip_path, "rb") as f:
            zip_bytes = f.read()

        # Ingest the raw data from the ZIP file
        df_raw = ingest_spotify_zip(zip_bytes)
        print("Raw Data Preview:")
        print(df_raw.head())
        print("\nRaw Data Info:")
        print(df_raw.info())

        # Clean the ingested data
        df_cleaned = clean_spotify_df(df_raw)
        print("\nCleaned Data Preview:")
        print(df_cleaned.head())
        print("\nCleaned Data Info:")
        print(df_cleaned.info())

import os
import sys
# Add the parent directory to the sys.path so that 'src' can be found.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap

# Import the ingestion and cleaning functions from the src folder
from src.data_ingestion import ingest_spotify_zip
from src.data_cleaning import clean_spotify_df

sns.set_style('whitegrid')

def load_data_from_zip(zip_bytes):
    """
    Reads the ZIP file bytes, ingests the JSON data,
    cleans the DataFrame, and returns it.
    """
    raw_df = ingest_spotify_zip(zip_bytes)
    if raw_df.empty:
        st.error("No JSON files found in the ZIP file.")
        return pd.DataFrame()
    cleaned_df = clean_spotify_df(raw_df)
    # Ensure the timestamp column is tz-naive for comparisons
    if 'ts' in cleaned_df.columns:
        if pd.api.types.is_datetime64tz_dtype(cleaned_df['ts']):
            cleaned_df['ts'] = cleaned_df['ts'].dt.tz_convert(None)
        else:
            cleaned_df['ts'] = pd.to_datetime(cleaned_df['ts'], errors='coerce')
    return cleaned_df

def main():
    st.title("My Personalized Spotify Wrapped Dashboard")
    
    st.write("Upload your ZIP file containing Spotify Streaming History JSON files. "
             "This app requires a ZIP file to display the analysis.")
    
    # 1. File Uploader: Load Data
    # ---------------------------
    uploaded_zip = st.file_uploader("Upload ZIP file", type="zip")
    
    # Require a ZIP file upload; no fallback to CSV
    if uploaded_zip is None:
        st.error("Please upload a ZIP file to display the dashboard.")
        return
    else:
        file_bytes = uploaded_zip.read()
        df = load_data_from_zip(file_bytes)
        if df.empty:
            st.error("The uploaded ZIP file did not contain any valid JSON data.")
            return

    # 2. Sidebar Filters
    # ------------------
    st.sidebar.header("Filter Options")
    
    # Filter 2.1: Playback Threshold
    threshold_seconds = st.sidebar.number_input("Ignore plays under (seconds)", min_value=0, value=20)
    threshold_ms = threshold_seconds * 1000
    if 'ms_played' in df.columns:
        df = df[df['ms_played'] >= threshold_ms]

    # Filter 2.2: Year Filters and Custom Date Range
    if 'ts' in df.columns:
        # Ensure the 'ts' column is tz-naive
        if pd.api.types.is_datetime64tz_dtype(df['ts']):
            df['ts'] = df['ts'].dt.tz_convert(None)
        df['year'] = df['ts'].dt.year
        unique_years = sorted(df['year'].dropna().unique())
        st.sidebar.subheader("Year Filters")
        selected_years = []
        for y in unique_years:
            if st.sidebar.checkbox(str(y), value=True):
                selected_years.append(y)
        if selected_years:
            df = df[df['year'].isin(selected_years)]
            
        st.sidebar.subheader("Custom Date Range")
        min_date = df['ts'].min().date()
        max_date = df['ts'].max().date()
        start_date = st.sidebar.date_input("Start Date", min_date)
        end_date = st.sidebar.date_input("End Date", max_date)
        df = df[(df['ts'] >= pd.to_datetime(start_date)) & (df['ts'] <= pd.to_datetime(end_date))]
    else:
        st.sidebar.write("Timestamp column not found.")

    # Filter 2.3: Artist Filter Mode and Selection
    st.sidebar.subheader("Artist Filter Mode")
    filter_mode = st.sidebar.radio("Select Mode", ["Include", "Exclude"], index=0)
    if 'artist_name' in df.columns:
        unique_artists = sorted(df['artist_name'].dropna().unique())
        st.sidebar.subheader("Artist Filter")
        selected_artists = st.sidebar.multiselect("Select Artists", unique_artists)
        if selected_artists:
            if filter_mode == "Include":
                df = df[df['artist_name'].isin(selected_artists)]
            else:
                df = df[~df['artist_name'].isin(selected_artists)]
    else:
        st.sidebar.write("Artist column not found.")

    # 3. Main Dashboard Content: Data Overview
    # -----------------------------------------
    st.subheader("Filtered Data Overview")
    st.write(f"Total records: {len(df):,}")
    private_cols = ['ip_addr', 'offline_timestamp']
    display_df = df.drop(columns=[col for col in private_cols if col in df.columns], errors='ignore')
    st.write(display_df.head())

    # 4. Overall Listening Stats
    # ----------------------------
    if 'ms_played' in df.columns:
        total_minutes = df['ms_played'].sum() / 60000.0
        st.subheader("Overall Listening Stats")
        st.write(f"Total Minutes Played: {total_minutes:,.2f}")
    else:
        st.write("Column 'ms_played' not found.")

    # 5. Top 10 Songs by Playback Count
    # ----------------------------------
    st.subheader("Top 10 Songs by Playback Count")
    if 'track_name' in df.columns:
        top_tracks_count = df['track_name'].value_counts().head(10)
        if 'artist_name' in df.columns and 'album_name' in df.columns:
            artist_album_map = df.groupby('track_name').agg({
                'artist_name': 'first',
                'album_name': 'first'
            })
        else:
            artist_album_map = pd.DataFrame(index=top_tracks_count.index)
            artist_album_map['artist_name'] = ''
            artist_album_map['album_name'] = ''
        top_songs_df = top_tracks_count.reset_index()
        top_songs_df.columns = ['track_name', 'count']
        top_songs_df = top_songs_df.join(artist_album_map, on='track_name')
        top_songs_df = top_songs_df.reset_index(drop=True)
        st.table(top_songs_df)

        st.subheader("Bar Chart of Top 10 Songs by Count")
        sorted_tracks = top_songs_df.sort_values('count', ascending=False).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.barh(sorted_tracks['track_name'], sorted_tracks['count'], color='#1DB954', edgecolor='none')
        ax.set_title("Top 10 Songs by Count", color='white', pad=10)
        ax.set_xlabel("Count", color='white')
        ax.set_ylabel("Track Name", color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        st.pyplot(fig)
    else:
        st.write("Column 'track_name' not found.")

    # 6. Top 10 Artists by Playback Count
    # ------------------------------------
    st.subheader("Top 10 Artists by Playback Count")
    if 'artist_name' in df.columns and 'ms_played' in df.columns:
        artist_stats = df.groupby('artist_name').agg(
            count=('artist_name', 'count'),
            total_ms_played=('ms_played', 'sum')
        ).reset_index()
        artist_stats['total_minutes'] = artist_stats['total_ms_played'] / 60000.0
        top_artists_df = artist_stats.sort_values('count', ascending=False).head(10).reset_index(drop=True)
        top_artists_df['count'] = top_artists_df['count'].apply(lambda x: f"{x:,}")
        top_artists_df['total_minutes'] = top_artists_df['total_minutes'].apply(lambda x: f"{round(x):,}")
        st.table(top_artists_df[['artist_name', 'count', 'total_minutes']])

        st.subheader("Bar Chart of Top 10 Artists by Count")
        sorted_artists = artist_stats.sort_values('count', ascending=False).head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.barh(sorted_artists['artist_name'], sorted_artists['count'], color='#1DB954', edgecolor='none')
        ax.set_title("Top 10 Artists by Count", color='white', pad=10)
        ax.set_xlabel("Count", color='white')
        ax.set_ylabel("Artist Name", color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        st.pyplot(fig)
    else:
        st.write("Column 'artist_name' not found.")

    # 7. Top 10 Albums with Expandable Details
    # -----------------------------------------
    st.subheader("Top 10 Albums by Playback Count (Expandable)")
    if 'album_name' in df.columns and 'track_name' in df.columns and 'artist_name' in df.columns:
        album_stats = df.groupby('album_name').agg(
            count=('album_name', 'count'),
            artist=('artist_name', 'first')
        ).reset_index()
        top_albums = album_stats.sort_values('count', ascending=False).head(10)
        for _, row in top_albums.iterrows():
            album = row['album_name']
            count = row['count']
            artist = row['artist']
            expander_label = f"{artist} - {album} — {count:,} plays"
            with st.expander(expander_label):
                album_df = df[df['album_name'] == album]
                track_counts = album_df['track_name'].value_counts()
                track_counts_df = track_counts.reset_index()
                track_counts_df.columns = ['track_name', 'count']
                st.table(track_counts_df)
    else:
        st.write("Required columns ('album_name', 'track_name', 'artist_name') not found.")

    # 8. Most and Least Skipped Songs Analysis
    # -----------------------------------------
    st.subheader("Most and Least Skipped Songs Analysis")
    if 'track_name' in df.columns and 'skipped' in df.columns:
        track_skip_stats = df.groupby('track_name').agg(
            total_plays=('track_name', 'count'),
            total_skips=('skipped', 'sum')
        ).reset_index()
        track_skip_stats['skip_rate'] = track_skip_stats['total_skips'] / track_skip_stats['total_plays']
        track_skip_stats = track_skip_stats[track_skip_stats['total_plays'] >= 5]
        
        most_skipped = track_skip_stats.sort_values(['skip_rate', 'total_plays'], ascending=[False, False]).head(10).reset_index(drop=True)
        least_skipped = track_skip_stats.sort_values(['skip_rate', 'total_plays'], ascending=[True, False]).head(10).reset_index(drop=True)
        
        if 'artist_name' in df.columns:
            artist_map = df.groupby('track_name')['artist_name'].first().reset_index()
            most_skipped = most_skipped.merge(artist_map, on='track_name', how='left')
            least_skipped = least_skipped.merge(artist_map, on='track_name', how='left')
        
        most_skipped = most_skipped[['artist_name', 'track_name', 'total_plays', 'total_skips', 'skip_rate']]
        least_skipped = least_skipped[['artist_name', 'track_name', 'total_plays', 'total_skips', 'skip_rate']]
        
        most_skipped['skip_rate'] = most_skipped['skip_rate'].apply(lambda x: f"{x:.2%}")
        least_skipped['skip_rate'] = least_skipped['skip_rate'].apply(lambda x: f"{x:.2%}")
        most_skipped['total_plays'] = most_skipped['total_plays'].apply(lambda x: f"{x:,}")
        least_skipped['total_plays'] = least_skipped['total_plays'].apply(lambda x: f"{x:,}")
        
        st.subheader("Top 10 Most Skipped Songs (by Skip Rate & Total Plays)")
        st.table(most_skipped.reset_index(drop=True))
        
        st.subheader("Top 10 Least Skipped Songs (by Skip Rate & Total Plays)")
        st.table(least_skipped.reset_index(drop=True))
    else:
        st.write("Columns 'track_name' or 'skipped' not found.")

    # 9. Monthly Top Artist Analysis
    # ------------------------------
    st.subheader("Monthly Top Artist Analysis")
    if 'artist_name' in df.columns and 'ms_played' in df.columns and 'track_name' in df.columns:
        df['year_month'] = df['ts'].dt.to_period('M')
        monthly_artist = df.groupby(['year_month', 'artist_name']).agg(
            total_ms_played=('ms_played', 'sum'),
            song_count=('track_name', 'count')
        ).reset_index()
        monthly_top = monthly_artist.sort_values(['year_month', 'total_ms_played'], ascending=[True, False]).drop_duplicates(subset=['year_month'], keep='first')
        top_artist_months = monthly_top.groupby('artist_name').agg(
            months_as_top=('year_month', 'count'),
            total_ms_played=('total_ms_played', 'sum'),
            total_song_count=('song_count', 'sum')
        ).reset_index()
        top_artist_months['total_minutes'] = top_artist_months['total_ms_played'] / 60000.0
        top_artist_months['total_minutes'] = top_artist_months['total_minutes'].round().astype(int)
        top_artist_months['months_as_top'] = top_artist_months['months_as_top'].astype(int)
        top_artist_months = top_artist_months.sort_values('months_as_top', ascending=False).reset_index(drop=True)
        st.table(top_artist_months[['artist_name', 'months_as_top', 'total_song_count', 'total_minutes']])
        
        st.subheader("Bar Chart: Top Artists by Months as Top")
        sorted_top_artist = top_artist_months.sort_values('months_as_top', ascending=False).head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.barh(sorted_top_artist['artist_name'], sorted_top_artist['months_as_top'], color='#1DB954', edgecolor='none')
        ax.set_title("Top Artists by Months as Top", color='white', pad=10)
        ax.set_xlabel("Months as Top", color='white')
        ax.set_ylabel("Artist Name", color='white')
        ax.tick_params(axis='x', colors='white', labelsize=10)
        ax.tick_params(axis='y', colors='white', labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        st.pyplot(fig)
    else:
        st.write("Required columns for Monthly Top Artist Analysis not found.")

    # 10. Listening Analysis: Graphs
    # ------------------------------
    st.subheader("Listening Analysis")
    
    # 10.1 Listening Time by Hour of Day
    st.subheader("Listening Time by Hour of Day")
    df['hour'] = df['ts'].dt.hour
    hourly_usage = df.groupby('hour')['ms_played'].sum() / 60000.0  # convert ms to minutes
    hourly_usage = hourly_usage.reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.bar(hourly_usage.index, hourly_usage.values, color='#1DB954', edgecolor='none')
    ax.set_title("Listening Time by Hour of Day", color='white', pad=10)
    ax.set_xlabel("Hour of Day", color='white')
    ax.set_ylabel("Total Minutes Played", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.set_xticks(range(24))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    st.pyplot(fig)

    # 10.2 Listening Time by Day of Week
    st.subheader("Listening Time by Day of Week")
    df['day_of_week'] = df['ts'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_usage = df.groupby('day_of_week')['ms_played'].sum() / 60000.0
    dow_usage = dow_usage.reindex(day_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.bar(dow_usage.index, dow_usage.values, color='#1DB954', edgecolor='none')
    ax.set_title("Listening Time by Day of Week", color='white', pad=10)
    ax.set_xlabel("Day of Week", color='white')
    ax.set_ylabel("Total Minutes Played", color='white')
    ax.tick_params(axis='x', colors='white', rotation=45)
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    st.pyplot(fig)

    # 10.3 Heatmap: Daily Listening by Hour and Day
    st.subheader("Heatmap: Daily Listening by Hour and Day")
    heatmap_df = df.copy()
    heatmap_df['hour'] = heatmap_df['ts'].dt.hour
    heatmap_df['day_of_week'] = heatmap_df['ts'].dt.day_name()
    grouped = heatmap_df.groupby(['hour', 'day_of_week'])['ms_played'].sum().reset_index()
    heatmap_pivot = grouped.pivot(index='hour', columns='day_of_week', values='ms_played') / 60000.0
    heatmap_pivot = heatmap_pivot.reindex(columns=day_order)
    min_minutes_played = heatmap_pivot.min().min()
    heatmap_pivot = heatmap_pivot.fillna(min_minutes_played)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    cmap = LinearSegmentedColormap.from_list("custom_green_white", ["#FFFFFF", "#1DB954"], N=256)
    sns_heatmap = sns.heatmap(
        heatmap_pivot,
        cmap=cmap,
        annot=False,
        fmt=".0f",
        ax=ax,
        linewidths=0,
        linecolor='none',
        cbar_kws={'label': 'Minutes Played'}
    )
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(colors='white')
    colorbar.set_label('Minutes Played', color='white')
    ax.set_title("Heatmap: Daily Listening by Hour and Day", color='white', pad=10)
    ax.set_xlabel("Day of Week", color='white')
    ax.set_ylabel("Hour of Day", color='white')
    ax.tick_params(axis='x', colors='white', rotation=45)
    ax.tick_params(axis='y', colors='white')
    st.pyplot(fig)

if __name__ == "__main__":
    main()

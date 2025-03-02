"""
Copyright (c) Nolichucky Associates 2024. All Rights Reserved.

This software is the confidential and proprietary information of Nolichucky Associates.
You shall not disclose such Confidential Information and shall use it only in accordance
 with the terms of the license agreement you entered into with Nolichucky Associates.

Unauthorized copying of this file, via any medium, is strictly prohibited.
Proprietary and confidential.

Project: Music Worcester Patron and Event Analytics

Author: Anthony Smith
Date: September, 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from timeit import default_timer as timer
import hashlib
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import requests
from requests.exceptions import RequestException
from sklearn.ensemble import IsolationForest

from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
import scipy.cluster.hierarchy as sch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def load_anonymized_dataset(anon_data_file, logger):
    start = timer()

    # Load event manifest file and fix column names
    event_df = pd.read_csv(anon_data_file)

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Anon Dataset loaded. Execution Time: {formatted_timing}')

    return event_df

"""
Function: calculate_genre_scores

Description:
    This function calculates genre preference scores for each account based on event attendance data. 
    It normalizes genre frequencies by adjusting the counts of events attended per genre to account 
    for the global frequency of each genre, giving higher weight to less frequent genres. 

Parameters:
    df (pd.DataFrame): A DataFrame containing event data with columns 'ContactId', 'EventDate', and 'EventGenre'.
    logger (logging.Logger): A logger object for logging execution details.

Process:
    1. Drop duplicates to obtain unique events per account, event date, and genre.
    2. Calculate the global frequency of each genre across all events.
    3. Calculate the count of events attended per genre for each account.
    4. Normalize the event counts by adjusting for the global frequency of the genre, giving less frequent genres higher weight.
    5. Compute the total normalized event count per account.
    6. For each account, calculate the normalized percentage of events attended for each genre.
    7. Reshape the data into a pivot table, where each row corresponds to an account and columns represent genres with their normalized scores.

Returns:
    A pivoted DataFrame where each row is an account and each column represents a genre's normalized percentage score.
"""
def calculate_genre_scores(df, logger):
    start = timer()

    # Drop duplicates to get unique events per ContactId, Event Date, and genre
    unique_events_df = df.drop_duplicates(subset=['ContactId', 'EventDate', 'EventGenre'])

    # Calculate the global frequency for each genre
    global_genre_freq = unique_events_df['EventGenre'].value_counts() / len(unique_events_df)

    # Calculate the by-genre counts for each account
    genre_counts = unique_events_df.groupby(['ContactId', 'EventGenre']).size().reset_index(name='Count')

    # Adjusted normalization: Normalize counts based on a modified factor
    genre_counts['NormalizedCount'] = genre_counts.apply(lambda row: row['Count'] / (1 + global_genre_freq[row['EventGenre']]), axis=1)

    # Calculate the total normalized count for each account
    total_counts = genre_counts.groupby('ContactId')['NormalizedCount'].sum().reset_index(name='TotalNormalized')

    # Calculate normalized percentage for each genre for each account
    genre_counts = genre_counts.merge(total_counts, on='ContactId')
    genre_counts['NormalizedPercentage'] = genre_counts['NormalizedCount'] / genre_counts['TotalNormalized']

    # Reshape and Finalize Data
    genre_df = genre_counts.pivot(index='ContactId', columns='EventGenre', values='NormalizedPercentage').fillna(0).reset_index()
    # Determine the preferred genre
    def get_preferred_genre(row):
        return row.idxmax()

    genre_df['PreferredGenre'] = genre_df.drop(columns=['ContactId']).apply(get_preferred_genre, axis=1)

    # Add a suffix of 'Score' to EventGenre columns
    genre_df.columns = [col + 'Score' if col != 'ContactId' and col != 'PreferredGenre' else col for col in genre_df.columns]


    # Define your is_omni function
    def is_omni(row, threshold=0.2, min_genres=3):
        return (row >= threshold).sum() >= min_genres

    # Get only the columns that end with "Score"
    score_columns = [col for col in genre_df.columns if col.endswith('Score')]

    # Apply the function only to these columns
    genre_df['Omni'] = genre_df[score_columns].apply(lambda row: is_omni(row), axis=1)

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Genre Scores complete. Execution Time: {formatted_timing}')

    return genre_df
"""
Function: calculate_rfm

Description:
    This function calculates Recency, Frequency, and Monetary (RFM) scores for each account based on their event participation 
    and purchasing behavior. It also computes related metrics such as days since the first, latest, and penultimate events. 
    The function normalizes the recency, frequency, and monetary values into a scoring system for further analysis and segmentation.

Parameters:
    df (pd.DataFrame): A DataFrame containing customer event and purchase data, including columns like 'ContactId', 'LatestEventDate', 'FirstEventDate', 'PenultimateEventDate', 'EventName', 'Quantity', and 'ItemPrice'.
    logger (logging.Logger): A logger object for logging execution details, debugging information, and completion time.

Process:
    1. Calculates the number of days since the latest, first, and penultimate events (setting negative values to 0 where relevant).
    2. Computes the frequency of distinct events attended by each account.
    3. Calculates monetary value as the product of item price and quantity for each account.
    4. Aggregates these metrics per account to get minimum recency, maximum frequency, and total monetary values.
    5. Segments Recency, Frequency, and Monetary values into score ranges (on a scale of 0 to 5) using predefined bins.
    6. Combines the individual RFM scores into a total RFM score for each account.
    7. Handles NaN values and accounts for customers who predate sales history.
    8. Logs the shape of the resulting RFM DataFrame, as well as execution time.

Returns:
    pd.DataFrame: A DataFrame containing calculated RFM scores and other aggregated customer data, with columns for RecencyScore, FrequencyScore, MonetaryScore, and the combined RFMScore.
"""
def calculate_rfm(df, binning, logger):
    start = timer()
    # Calculate today's date or a specific snapshot date
    today = datetime.today() # or replace with a specific date

    # Calculate days from today for both CreatedDate and LatestEventDate
    df['DaysFromLatestEvent'] = (today - df['LatestEventDate']).dt.days

    # Calculate Recency as the maximum of DaysFromCreated or DaysFromLatestEvent,
    # limited to a minimum of 0. Future events are Recency = 0.
    # Given our season periods, recency of days beyond one season can be tricky.
    # TODO: Add "Season recency" as a better alternative?
    df['Recency'] = df['DaysFromLatestEvent']
    df['Recency'] = np.where(df['Recency'] < 0, 0, df['Recency'])

    # Calculate days since first event, clipping to 0 if in the future
    df['DaysFromFirstEvent'] = (today - df['FirstEventDate']).dt.days
    df['DaysFromFirstEvent'] = np.where(df['FirstEventDate'] > today, 0, df['DaysFromFirstEvent'])

    # Calculate days since penultimate event, setting to zero if in the future or NA
    # We need this to determine "Returning" patrons, especially given the pandemic lull.
    df['DaysFromPenultimateEvent'] = (today - df['PenultimateEventDate']).dt.days
    df['DaysFromPenultimateEvent'] = np.where(df['PenultimateEventDate'] > today, 0, df['DaysFromPenultimateEvent'])
    df['DaysFromPenultimateEvent'].fillna(0, inplace=True)

    # Frequency = Count of distinct events attended
    # TODO: Again, perhaps Season Frequency would be more useful.
    df['Frequency'] = df.groupby('ContactId')['EventName'].transform('nunique')

    # Monetary = Quantity * ItemPrice
    # Monetary is a bit dubious because it doesn't include donations,
    # and ticket prices have varied so much over time. It
    # 's mostly used to exclude pure comp ticket patrons.
    df['Monetary'] = df['Quantity'] * df['ItemPrice']

    # Calculate RFM values
    rfm_df = df.groupby('ContactId').agg({
        'Recency': 'min',
        'Frequency': 'max',
        'Monetary': 'sum',
        'DaysFromFirstEvent': 'min',
        'DaysFromPenultimateEvent': 'min',
        'Subscriber': 'last',
        'ChorusMember': 'last',
        'DuesTxn': 'last',
        'FrequentBulkBuyer': 'last',
        'Student': 'last'
    })
    logger.info(rfm_df.shape)

    # Fill NaN values
    rfm_df['Recency'].fillna(0, inplace=True)
    rfm_df['Frequency'].fillna(0, inplace=True)
    rfm_df['Monetary'].fillna(0, inplace=True)
    rfm_df['DaysFromFirstEvent'].fillna(3000, inplace=True) # if no DaysFromFirstEvent, then very large

    if binning == 0: 
        # bin boundaries are subjectively based on the data set and an understanding of the business and patron base.
        # A clustering approach would be better
        bins = [0, 120, 400, 700, 1400, 2000, float('inf')]
        labels = [5, 4, 3, 2, 1, 0]
        rfm_df['RecencyScore'] = pd.cut(rfm_df['Recency'], bins=bins, labels=labels, right=False).astype(int)
        logger.debug(f'Recency Score OK')

        bins = [0,1, 2, 4, 8, 12, float('inf')]
        labels = [0, 1, 2, 3, 4, 5]
        rfm_df['FrequencyScore'] = pd.cut(rfm_df['Frequency'], bins=bins, labels=labels, right=False).astype(int)
        logger.debug(f'Frequency Score OK')

        bins = [0, 10, 70, 200, 400, 1000, float('inf')]
        labels = [0, 1, 2, 3, 4, 5]
        rfm_df['MonetaryScore'] = pd.cut(rfm_df['Monetary'], bins=bins, labels=labels, right=False)

        # extra step to handle outlier patrons who predate sales history.
        rfm_df['MonetaryScore'] = rfm_df['MonetaryScore'].fillna(0)
        # Now safely convert to int
        rfm_df['MonetaryScore'] = rfm_df['MonetaryScore'].astype(int)

        logger.debug(f'Monetary Score OK')

        # Aggregate scores
        rfm_df['RFMScore'] = rfm_df['RecencyScore'] + rfm_df['FrequencyScore'] + rfm_df['MonetaryScore']
        rfm_df['RFMScore'] = rfm_df['RFMScore'].fillna(0)
        
    elif binning == 1: #Aayan
        rfm_df['RecencyScore'] = pd.qcut(rfm_df['Recency'], 6, labels=[5, 4, 3, 2, 1, 0])
        logger.debug(f'Recency Score OK')
        rfm_df['FrequencyScore'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 6, labels=[0, 1, 2, 3, 4, 5])
        logger.debug(f'Frequency Score OK')
        rfm_df['MonetaryScore'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 6, labels=[0, 1, 2, 3, 4, 5])
        logger.debug(f'Monetary Score OK')
        rfm_df['RFMScore'] = (rfm_df['RecencyScore'].astype(int) + 
                            rfm_df['FrequencyScore'].astype(int) + 
                            rfm_df['MonetaryScore'].astype(int))
        rfm_df['RFMScore'].fillna(0, inplace=True)
    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'RFM scores complete. Execution Time: {formatted_timing}')

    return rfm_df
"""
Function: assign_segment

Description:
    This function assigns a customer to a specific segment based on Recency, Frequency, and Monetary (RFM) scores, as 
    well as other behavioral and engagement metrics. It categorizes customers into meaningful segments for targeted marketing, 
    customer retention, and engagement strategies.

Parameters:
    df (pd.Series): A row of customer data, containing various columns like 'RecencyScore', 'FrequencyScore', 
    'MonetaryScore', 'Subscriber', 'DaysFromFirstEvent', 'DaysFromPenultimateEvent', etc.
    new_threshold (int): The maximum number of days from the first event to qualify a customer as "New".
    returning_threshold (int): The maximum number of days since the penultimate event to qualify a customer as "Returning".

Process:
    1. First, the function prunes out specific groups, such as:
        - 'Comp': Customers with a monetary score of 0.
        - 'Group Buyer': Frequent bulk buyers.
        - 'New': Customers whose first event was recent, based on the new_threshold.
        - 'Returning': Customers who attended an event recently but had a long gap before the previous event.
        - 'Best': Customers with the highest possible RFM score (15).
    
    2. If not pruned, it checks for high engagement:
        - 'Potential Chorus Subscriber' or 'Potential Subscriber': Based on recency, frequency, and chorus membership.
        - 'High': High engagement with frequent and recent event attendance.
    
    3. Then, other segments are determined based on the customer's recency and frequency scores:
        - 'Upsell', 'Reminder', 'Come Again', 'Lapsed', 'One&Done', 'Slipping'.
    
    4. If no other conditions are met, the customer is assigned to the 'Others' segment.

Returns:
    str: The assigned segment as a string value, representing the category into which the customer falls.
"""
def assign_segment(df, new_threshold, returning_threshold):

    # Helper functions for common checks
    def is_potential_subscriber():
        return df['RecencyScore'] >= 4 and df['Subscriber'] != 'True'

    def is_high_engagement():
        return df['RecencyScore'] >= 4 and df['FrequencyScore'] >= 4

    # Prune out specific groups first
    if df['MonetaryScore'] == 0:
        return 'Comp'
    if df['FrequentBulkBuyer']:
        return 'Group Buyer'
    if df['DaysFromFirstEvent'] <= new_threshold:
        return 'New'
    if df['RecencyScore'] > 4 and df['DaysFromPenultimateEvent'] > returning_threshold:
        return 'Returning'
    if df['RFMScore'] == 15:
        return 'Best'

    # Check for high engagement and subscriber potential
    if is_high_engagement():
        if df['ChorusMember'] == 'True' and df['Subscriber'] != 'True':
            return 'Potential Chorus Subscriber'
        if is_potential_subscriber():
            return 'Potential Subscriber'
        return 'High'

    # Other segments based on Recency and Frequency
    if df['RecencyScore'] >= 3:
        if df['FrequencyScore'] >= 2:
            return 'Upsell'
        return 'Come Again' if df['FrequencyScore'] > 1 else 'Reminder'

    if df['RecencyScore'] < 2:
        return 'One&Done' if df['FrequencyScore'] <= 1 else 'Lapsed'

    if df['RecencyScore'] <= 3 and df['FrequencyScore'] >= 2:
        return 'Slipping'

    return 'Others'
#----------------------------------------------------------------------
def calculate_regularity(data, logger=None):
    """
    Calculate Regularity based on their event attendance data. It combines inter-season,
    intra-season, average event frequency per season, and event "cluster attendance. 
    We filter only for events and seasons in each patron's actual attendance period.
    """
    datamerge = r"C:\Users\akris\OneDrive\Desktop\Music Worcester Work\anon_DataMerge.csv"
    df = pd.read_csv(datamerge,low_memory=False)


    today = datetime.today()
    df['EventDate'] = pd.to_datetime(df['EventDate'], errors='coerce')
    # Determine FiscalYear for season-based recency calculation
    df['FiscalYear'] = df['EventDate'].apply(lambda x: x.year if x.month > 6 else x.year - 1)

    # Determine the first eligible event and fiscal year for each patron
    df['FirstEventDate'] = df.groupby('ContactId')['EventDate'].transform('min')
    df['FirstFiscalYear'] = df.groupby('ContactId')['FiscalYear'].transform('min')

    # Filter out events and seasons not eligible for each patron
    df = df[df['EventDate'] >= df['FirstEventDate']]

    # Calculate SeasonRecency as the difference between the current year and the fiscal year
    df['SeasonRecency'] = today.year - df['FiscalYear']

    # Sort by ContactId and EventDate to calculate the gap between consecutive events
    df = df.sort_values(by=['ContactId', 'EventDate'])

    # Calculate the difference in days between consecutive events for each patron
    df['EventGap'] = df.groupby('ContactId')['EventDate'].diff().dt.days

    # Cluster events that occur within 4 consecutive days (e.g., weekend festivals)
    #df['EventCluster'] = (df['EventGap'] > 3).cumsum()  # New cluster if gap > 4 days
    df['EventCluster'] = (df.groupby('ContactId')['EventDate']
                          .transform(lambda x: (x.diff().dt.days > 3).cumsum()))

    # Calculate ClusterFrequency for each account by counting unique clusters within each season
    df['ClusterFrequency'] = df.groupby(['ContactId', 'FiscalYear'])['EventCluster'].transform('nunique')

    # Calculate SeasonCount (number of unique seasons attended) for inter-season regularity
    df['SeasonCount'] = df.groupby('ContactId')['FiscalYear'].transform('nunique')

    # Calculate EventFrequencyPerSeason (average events attended per season)
    total_events = df.groupby('ContactId')['EventCluster'].transform('count')
    df['EventFrequencyPerSeason'] = total_events / df['SeasonCount']

    # Define max values for normalization of metrics
    max_season_count = df['SeasonCount'].max()
    max_cluster_frequency = df['ClusterFrequency'].max()
    max_event_frequency_per_season = df['EventFrequencyPerSeason'].max()

    # Calculate the final Regularity score with weights for each metric
    w1, w2, w3 = 0.4, 0.2, 0.4

    # Regularity score calculation
    df['Regularity'] = (w1 * (df['SeasonCount'] / max_season_count) +
                        w2 * (df['ClusterFrequency'] / max_cluster_frequency) +
                        w3 * (df['EventFrequencyPerSeason'] / max_event_frequency_per_season))
    
    print("Dataset with RegionAssignment (AccountId):\n", data.head())
    print("Dataset without RegionAssignment (ContactId):\n", df.head())

    merged_dataset = pd.merge(data, 
                          df[['ContactId', 'Regularity']], 
                          on='ContactId', how='left')
    print("Merged Dataset with 'RegionAssignment' column added:\n", merged_dataset.head())

    return merged_dataset

def IsoForest(data, years, contamination=0.05, random_state=42):
    data['FirstEventDate'] = pd.to_datetime(data['FirstEventDate'])
    data['DaysFromFirstEvent'] = (datetime.now() - data['FirstEventDate']).dt.days

    data['Recency (Months)'] = data['Recency (Months)'] < (years * 12)
    data['AUM'] = data.apply(lambda row: row['Monetary'] / row['DaysFromFirstEvent'] if row['DaysFromFirstEvent'] > 0 else 0, axis=1)
    data["AYMGSForcast"] = data["AYM_Z"] + data["GrowthScore_Z"]
    data['EngagementConsistency'] =  data.apply(lambda row: row['Frequency'] / row['DaysFromFirstEvent'] if row['DaysFromFirstEvent'] > 0 else 0, axis=1)

    rfm_data = data[['Lifespan', 'PreferenceConfidence', 'AYMGSForcast', 'Regularity']].copy()

    rfm_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    rfm_data.fillna(0, inplace=True)

    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outliers = iso_forest.fit_predict(rfm_data)
    data['Outlier'] = outliers
    inliers_df = data[data['Outlier'] == 1].drop(columns='Outlier')
    outliers_df = data[data['Outlier'] == -1]
    outlier_count = len(outliers_df)
    print(f"Total Outliers Detected: {outlier_count}")
    return inliers_df, outlier_count


def analyze_silhouette_scores(df, max_clusters=10): #AAYAN - displays the silhouette plot for k means
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    silhouette_avg_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(df[['Lifespan', 'PreferenceConfidence', 'AYMGSForcast', 'Regularity']])
        silhouette_avg = silhouette_score(df[['Lifespan', 'PreferenceConfidence', 'AYMGSForcast', 'Regularity']], cluster_labels)
        silhouette_avg_scores.append(silhouette_avg)
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_avg_scores, marker='o', linestyle='--', color='skyblue', label='Average Silhouette Score')
    plt.title('Silhouette Scores for Varying Number of Clusters', fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('Average Silhouette Score', fontsize=14)
    plt.xticks(range(2, max_clusters + 1))  # Set x-ticks to match the range of clusters
    plt.grid(True)
    plt.axhline(y=max(silhouette_avg_scores), color='red', linestyle='--', label='Max Silhouette Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_silhouette_scores_agglomerative(df, max_clusters=10): #AAYAN - displays the silhouette scores for agglomerative
    silhouette_avg_scores = []
    rfm_aum_data = df[['Lifespan', 'PreferenceConfidence', 'AYMGSForcast', 'Regularity']].copy()
    rfm_aum_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    rfm_aum_data.fillna(0, inplace=True)
    scaler = StandardScaler()
    rfm_aum_scaled = scaler.fit_transform(rfm_aum_data)
    for n_clusters in range(2, max_clusters + 1):
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = agglomerative.fit_predict(rfm_aum_scaled)
        silhouette_avg = silhouette_score(rfm_aum_scaled, cluster_labels)
        silhouette_avg_scores.append(silhouette_avg)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_avg_scores, marker='o', linestyle='--', color='skyblue', label='Avg Silhouette Score')
    plt.title('Silhouette Scores for Varying Clusters (Agglomerative)', fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('Average Silhouette Score', fontsize=14)
    plt.grid(True)
    plt.axhline(y=max(silhouette_avg_scores), color='red', linestyle='--', label='Max Silhouette Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_AUM_and_KMeans(df, years):  # AAYAN - Calculates AUM and KMeans score
    df['Recency (Months)'] = df['Recency (Months)'] < (years * 12)
    df['AUM'] = df.apply(lambda row: row['Monetary'] / row['DaysFromFirstEvent'] if row['DaysFromFirstEvent'] > 0 else 0, axis=1)
    df["AYMGSForcast"] = df["AYM_Z"] + df["GrowthScore_Z"]
    df['EngagementConsistency'] =  df.apply(lambda row: row['Frequency'] / row['DaysFromFirstEvent'] if row['DaysFromFirstEvent'] > 0 else 0, axis=1)

    rfm_aum_data = df[['Lifespan', 'PreferenceConfidence', 'AYMGSForcast', 'Regularity']].copy()

    rfm_aum_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    rfm_aum_data.fillna(0, inplace=True)


    plt.figure(figsize=(10, 8))
    sns.heatmap(rfm_aum_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap of RFM and AUM Data")
    plt.show()
    
    scaler = StandardScaler()
    rfm_aum_scaled = scaler.fit_transform(rfm_aum_data)
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(rfm_aum_scaled)
        wcss.append(kmeans.inertia_)
    optimal_clusters = 4
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(rfm_aum_scaled)
    df['Cluster KMeans'] = kmeans.predict(rfm_aum_scaled) + 1
    return df, wcss, kmeans.cluster_centers_  

def calculate_centroids_kmeans(df):  # AAYAN - calculates the center of the Kmeans cluster
    centroids = df.groupby('Cluster KMeans')[['Lifespan', 'PreferenceConfidence', 'AYMGSForcast', 'Regularity']].mean()
    print(centroids)

def calculate_AUM_and_Agglomerative(df, years): # AAYAN - Calculates AUM and Agglomerative score
    df['Recency (Months)'] = df['Recency (Months)'] < (years * 12)
    df['AUM'] = df.apply(lambda row: row['Monetary'] / row['DaysFromFirstEvent'] if row['DaysFromFirstEvent'] > 0 else 0, axis=1)
    df["AYMGSForcast"] = df["AYM_Z"] + df["GrowthScore_Z"]
    df['EngagementConsistency'] =  df.apply(lambda row: row['Frequency'] / row['DaysFromFirstEvent'] if row['DaysFromFirstEvent'] > 0 else 0, axis=1)

    rfm_aum_data = df[['Lifespan', 'PreferenceConfidence', 'AYMGSForcast', 'Regularity']].copy()

    rfm_aum_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    rfm_aum_data.fillna(0, inplace=True)
    scaler = StandardScaler()
    rfm_aum_scaled = scaler.fit_transform(rfm_aum_data)
    optimal_clusters = 4
    agglomerative = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
    df['Cluster Agglo'] = agglomerative.fit_predict(rfm_aum_scaled) + 1

    return df  

def calculate_centroids_agglomerative(df):  #AAYAN - displays the centroids for agglomerative
    centroids = df.groupby('Cluster Agglo')[['Lifespan', 'PreferenceConfidence', 'AYMGSForcast', 'Regularity']].mean()
    print(centroids) 

def calculate_AUM_and_KMedoids(df, years):  # AAYAN - Calculates AUM and KMedoids score
    # Calculate AUM, EngagementConsistency, etc.
    df['Recency (Months)'] = df['Recency (Months)'] < (years * 12)
    df['AUM'] = df.apply(lambda row: row['Monetary'] / row['DaysFromFirstEvent'] if row['DaysFromFirstEvent'] > 0 else 0, axis=1)
    df["AYMGSForcast"] = df["AYM_Z"] + df["GrowthScore_Z"]
    df['EngagementConsistency'] =  df.apply(lambda row: row['Frequency'] / row['DaysFromFirstEvent'] if row['DaysFromFirstEvent'] > 0 else 0, axis=1)
    rfm_aum_data = df[['Lifespan', 'PreferenceConfidence', 'AYMGSForcast', 'Regularity']].copy()
    
    rfm_aum_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    rfm_aum_data.fillna(0, inplace=True)
   
    scaler = StandardScaler()
    rfm_aum_scaled = scaler.fit_transform(rfm_aum_data)

    silhouette_scores = []
    for n_clusters in range(2, 11):
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=0, metric='euclidean', init='k-medoids++')
        labels = kmedoids.fit_predict(rfm_aum_scaled)
        silhouette = silhouette_score(rfm_aum_scaled, labels)
        silhouette_scores.append(silhouette)

    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 3

    kmedoids = KMedoids(n_clusters=optimal_clusters - 1, random_state=0, metric='euclidean', init='k-medoids++')
    df['Cluster KMedoids'] = kmedoids.fit_predict(rfm_aum_scaled) + 1
    
    centroids = scaler.inverse_transform(kmedoids.cluster_centers_)
    return df,  silhouette_scores, centroids

def KMeans_ElbowPlot(wcss): #AAYAN - displays the elbow plot for k means
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b', label='WCSS')
    plt.title('Elbow Method for Optimal Clusters', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=14)
    plt.xlim(1, 10)
    plt.ylim(min(wcss) - 1000, max(wcss) + 1000)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray')
    for i, w in enumerate(wcss):
        plt.text(i + 1, w + 100, f'{w:.0f}', ha='center', va='bottom', fontsize=10, color='black')
    plt.tight_layout()
    plt.legend()
    plt.show()
    
def snake_plot_rfm(df, k_values=[4, 5, 6]): #AAYAN - displays the snake plot for k means
    rfm_aum_data = df[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    rfm_aum_scaled = scaler.fit_transform(rfm_aum_data)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        df['Cluster KMeans'] = kmeans.fit_predict(rfm_aum_scaled)
        cluster_means = df.groupby('Cluster KMeans')[['Recency', 'Frequency', 'Monetary']].mean().reset_index(drop=True)
        cluster_means_scaled = scaler.inverse_transform(cluster_means)
        plt.figure(figsize=(10, 6))
        for i in range(k):
            plt.plot(['Recency', 'Frequency', 'Monetary'], cluster_means_scaled[i], marker='o', label=f'Cluster {i+1}')
        plt.title(f'Snake Plot for K = {k}', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=14)
        plt.ylabel('Mean Values', fontsize=14)
        plt.xticks(['Recency', 'Frequency', 'Monetary'])  # Set x-ticks to match metric names
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def KMeans_RFMPlot(df): #AAYAN - displays the RFM plot for k means
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['Recency'], df['Frequency'], df['Monetary'], c=df['Cluster KMeans'], cmap='viridis', s=100, alpha=0.7)
    ax.set_title('3D K-means Clustering of RFM', fontsize=16, fontweight='bold')
    ax.set_xlabel('Recency', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_zlabel('Monetary', fontsize=14)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster', fontsize=12)
    plt.show()


def plot_dendrogram(df): #AAYAN - displays the dendrogram for agglomerative
    rfm_aum_data = df[['Recency', 'Frequency', 'Monetary', 'AUM', 'DaysFromPenultimateEvent']]
    scaler = StandardScaler()
    rfm_aum_scaled = scaler.fit_transform(rfm_aum_data)
    linkage_matrix = sch.linkage(rfm_aum_scaled, method='ward')
    plt.figure(figsize=(10, 7))
    plt.title('Dendrogram for Hierarchical Clustering')
    sch.dendrogram(linkage_matrix)
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()

def analyze_clusters_and_outliers(df, clustertype):
    cluster_counts = df[clustertype].value_counts()
    print("Cluster sizes:\n", cluster_counts)

    threshold = df.shape[0] * 0.05
    outlier_clusters = cluster_counts[cluster_counts < threshold].index
    
    outliers = df[df[clustertype].isin(outlier_clusters)]
    print("Outliers identified:\n", outliers)
    
    plt.figure(figsize=(10, 6))
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title('Cluster Sizes After Clustering')
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Data Points')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=12)
    plt.show()
    



def Agglomerative_RFMPlot(df): #AAYAN - displays the RFM plot for agglomerative
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['Recency'], df['Frequency'], df['Monetary'], c=df['Cluster Agglo'], cmap='viridis', s=100, alpha=0.7)
    ax.set_title('3D Agglomerative Clustering of RFM', fontsize=16, fontweight='bold')
    ax.set_xlabel('Recency', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_zlabel('Monetary', fontsize=14)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster', fontsize=12)
    plt.show()

def compare_clustering_solutions(agglomerative_df, kmeans_df):  #AAYAN - compares clustering sollutions for both
    comparison = pd.DataFrame({'Agglomerative_Cluster': agglomerative_df['Cluster Agglo'],'KMeans_Cluster': kmeans_df['Cluster KMeans']})
    crosstab = pd.crosstab(comparison['Agglomerative_Cluster'], comparison['KMeans_Cluster'])
    print(crosstab)
    return crosstab

def knn_classification_with_split(data, target_col, k_values=[1, 3, 5, 7, 9], test_size=0.2, random_state=42): #AAYAN - k_nn
    data = data.dropna(subset=[target_col])
    features = ['Recency', 'Frequency', 'Monetary','AUM', 'DaysFromPenultimateEvent', 'Cluster KMeans']
    X = data[features]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        confusion = confusion_matrix(y_test, predictions)
        results[k] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'confusion_matrix': confusion
        }
    for k in results:
        print(f'K={k}: Accuracy = {results[k]["accuracy"]}')
        print('Predictions:', results[k]['predictions'])
        print('Confusion Matrix:\n', results[k]['confusion_matrix'])


#AAYAN
# # # traindata = r'c:\Users\akris\OneDrive\Desktop\Music Worcester Work\Patrondetails_train_data.csv' 
# # # testdata = r'C:\Users\akris\OneDrive\Desktop\Music Worcester Work\Patrondetails_test_data.csv'
alldata = r"C:\Users\akris\OneDrive\Desktop\Music Worcester Work\anon_PatronDetails-2024.12.03 - anon_Patrons.csv.csv"
#sheetdata = r"C:\Users\akris\OneDrive\Desktop\Music Worcester Work\1219Clustering - LatestMetrics.csv"
traindata = pd.read_csv(alldata,low_memory=False)
# # # # traindata = calculate_regularity(traindata, logger=None)
# # # # # Identifying Outliers----------------------------------
traindata, outliers = IsoForest(traindata, 0.0, contamination=0.05, random_state=42)
# # # # # K-MeansClustering----------------------------------
kmeansdf, wcss, kmeans_centers = calculate_AUM_and_KMeans(traindata, 3.0)
print("Kmeans been calced")
# analyze_silhouette_scores(kmeansdf, max_clusters=10)
calculate_centroids_kmeans(kmeansdf)
# # # # KMeans_ElbowPlot(wcss)
# # # # # snake_plot_rfm(kmeansdf, k_values=[4, 5, 6])
# # # # # KMeans_RFMPlot(kmeansdf)
# # # # # AgglomerativeClustering---------------------------------
aggdf = calculate_AUM_and_Agglomerative(kmeansdf, 3.0)
print("Agglo been calced")
# analyze_silhouette_scores_agglomerative(traindata, max_clusters=10)
calculate_centroids_agglomerative(aggdf)
# # # # # plot_dendrogram(aggdf)
# # # # # analyze_clusters_and_outliers(kmeansdf, 'Cluster KMeans')
# # # # analyze_silhouette_scores_agglomerative(traindata, max_clusters=10)
# # # # # Agglomerative_RFMPlot(aggdf)
# # # # # compare_clustering_solutions(aggdf, kmeansdf)
# # # # # # KNN----------------------------------
# # # # # knn_classification_with_split(kmeansdf, 'RegionAssignment', k_values=[1, 3, 5, 7, 9], test_size=0.2, random_state=42)
# # # #Kmediod
df, silhouette_scores, centroids  = calculate_AUM_and_KMedoids(aggdf, 3.0)
print("Kmed been calced")
# analyze_clusters_and_outliers(df, 'Cluster KMeans')
print(silhouette_scores)
print(centroids)
df.to_excel(r"C:\Users\akris\OneDrive\Desktop\Music Worcester Work\1219Clustering - LatestMetrics.xlsx") 
print("its been outputted bruh")
analyze_clusters_and_outliers(df, 'Cluster KMeans')
analyze_clusters_and_outliers(df, 'Cluster Agglo')
analyze_clusters_and_outliers(df, 'Cluster KMedoids')




# General functions
def safe_divide(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(x, y)
        result[~np.isfinite(result)] = 0  # Set NaN, inf, -inf to 0
    return result

def plot_RFM(df, logger):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    xs = df['Recency']
    ys = df['Frequency']
    zs = df['Monetary']

    ax.scatter(xs, ys, zs)

    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
def plot_3D_scatter(xs,x_label, ys, y_label,zs, z_label, logger):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
def plot_2D_scatter(x,x_label, y, y_label, logger):
    plt.figure(figsize=(20, 20))
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Distribution of {x_label} and {y_label}')
    plt.show()

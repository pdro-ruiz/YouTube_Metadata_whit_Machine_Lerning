"""
utf-8
preprocess_data.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This module handles loading and preprocessing raw data stored in the `/data/raw_data` directory.
The preprocessing involves multiple steps, including text normalization, data extraction, feature engineering, text vectorization, and duplicate removal, among others.

Data from different regions is loaded from CSV and JSON files, processed individually, and merged.
Final processed data for each region is stored in `/data/data_processed` with appropriate names.
Once all regions are processed, the data is concatenated and stored in the same directory as `all_data_processed`.

Functions:
- prepare_files
- clean_data
- transform_text_columns
- feature_engineering
- vectorize_text_columns
- analyze_and_remove_duplicates
- preprocess_data
- main

Usage:
This script can be executed from the command line as follows:
-->    python main.py --config_path train.conf --input_dir data/raw_data --output_dir data/industrialized
"""


import os
import shutil
import re
import json
import logging
import pickle
import numpy as np
import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
from nltk.corpus import stopwords

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.propagate = False  # Avoid propagation to parent logger
    return logger


def prepare_files(input_dir, prep_dir):
    """
    Prepares CSV and JSON files for processing, encoding them and replacing line breaks in text fields. 
    Copies JSON files to the preprocessing directory.
    """
    os.makedirs(prep_dir, exist_ok=True)
    logger = logging.getLogger('preprocess_logger')

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(prep_dir, filename)
            for encoding in ['utf-8', 'ISO-8859-1', 'latin1']:
                try:
                    df = pd.read_csv(input_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError(f"Decoding error while processing: {filename}")
            df = df.replace('\n', ' ', regex=True).replace('\r', ' ', regex=True)
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(prep_dir, filename)
            shutil.copy(input_path, output_path)

    logger.info(f"Files prepared and saved in: {prep_dir}")

def clean_data(df):
    """
    Cleans the dataset by removing records with invalid video_id and unwanted categories.
    """
    logger = logging.getLogger('preprocess_logger')

    df_cleaned = df[df['video_id'] != '#NAME?']
    df_cleaned = df_cleaned[df_cleaned['category_id'] != 29]
    df_cleaned = df_cleaned[~df_cleaned['category_id'].isin([15, 19, 30, 43, 44])]

    logger.info("Character and category cleaning completed")

    return df_cleaned

def transform_text_columns(df):
    """
    Transforms the text columns: 
        - Converts to lowercase
        - Extracts emojis
        - Counts text length
        - Extracts URLs
        - Cleans text
        - Converts text columns to word lists.
    """
    logger = logging.getLogger('preprocess_logger')

    # Replace null values in 'description' before cleaning
    df['description'] = df['description'].fillna('unknown')


    # Emoji extractor
    def extract_emojis(text):
        if isinstance(text, str):
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"                 # Emoticons
                "\U0001F300-\U0001F5FF"                 # Symbols & pictographs
                "\U0001F680-\U0001F6FF"                 # Transport & map symbols
                "\U0001F1E0-\U0001F1FF"                 # Flags (iOS)
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE
            )
            return emoji_pattern.findall(text)
        return []

    # URL extractor
    def extract_urls(text):
        if isinstance(text, str):
            url_pattern = re.compile(r'http[s]?://\S+|www\.\S+')
            return url_pattern.findall(text)
        return []

    # Text cleaner
    def clean_text(text):
        if isinstance(text, str):
            text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
            text = re.sub(
                "["
                "\U0001F600-\U0001F64F"
                "\U0001F300-\U0001F5FF"
                "\U0001F680-\U0001F6FF"
                "\U0001F1E0-\U0001F1FF"
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+", '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return text.lower()
        return text

    # Convert text to word list
    def word_list(text):
        if isinstance(text, str):
            return text.split()
        return []

    # Transformations
    df['title_cleaned'] = df['title'].apply(clean_text)
    df['channel_title_cleaned'] = df['channel_title'].apply(clean_text)
    df['tags_cleaned'] = df['tags'].apply(clean_text)
    df['description_cleaned'] = df['description'].apply(clean_text)

    df['title_length'] = df['title_cleaned'].apply(len)
    df['description_length'] = df['description_cleaned'].apply(len)
    df['tags_count'] = df['tags_cleaned'].apply(lambda x: len(x.split()) if pd.notnull(x) else 0)

    df['emojis'] = df['title'].apply(extract_emojis) + df['description'].apply(extract_emojis)
    df['urls'] = df['description'].apply(extract_urls)

    df['title_words'] = df['title_cleaned'].apply(word_list)
    df['tags_words'] = df['tags_cleaned'].apply(word_list)
    df['description_words'] = df['description_cleaned'].apply(word_list)

    logger.info("Text column transformation completed")

    return df


def feature_engineering(df):
    """
    Performs feature engineering on the DataFrame, creating new features such as:
        - Date encodings in cyclic format
        - Ratios and logarithmic transformations
    """
    logger = logging.getLogger('preprocess_logger')

    # Cyclic date encoding
    df['publish_month_sin'] = np.sin(2 * np.pi * df['publish_month'] / 12)
    df['publish_month_cos'] = np.cos(2 * np.pi * df['publish_month'] / 12)
    df['trending_month_sin'] = np.sin(2 * np.pi * df['trending_month'] / 12)
    df['trending_month_cos'] = np.cos(2 * np.pi * df['trending_month'] / 12)
    df['publish_day_sin'] = np.sin(2 * np.pi * df['publish_day'] / 31)
    df['publish_day_cos'] = np.cos(2 * np.pi * df['publish_day'] / 31)
    df['publish_hour_sin'] = np.sin(2 * np.pi * df['publish_hour'] / 24)
    df['publish_hour_cos'] = np.cos(2 * np.pi * df['publish_hour'] / 24)
    df['trending_day_of_week_sin'] = np.sin(2 * np.pi * df['trending_day_of_week'] / 7)
    df['trending_day_of_week_cos'] = np.cos(2 * np.pi * df['trending_day_of_week'] / 7)

    # Ratios and logarithmic transformations
    df['days_to_trend'] = (df['trending_date'] - pd.to_datetime(df['publish_date'])).dt.days
    df['like_dislike_ratio'] = df['likes'] / (df['dislikes'] + 1)
    df['like_view_ratio'] = df['likes'] / (df['views'] + 1)
    df['comment_view_ratio'] = df['comment_count'] / (df['views'] + 1)
    df['log_views'] = np.log1p(df['views'])
    df['log_likes'] = np.log1p(df['likes'])
    df['log_dislikes'] = np.log1p(df['dislikes'])
    df['log_comment_count'] = np.log1p(df['comment_count'])

    logger.info("Feature engineering completed")

    return df


# Download stopwords if not available
nltk.download('stopwords')  # Make sure to download stopwords
stop_words_multilingual = {
    'en': set(stopwords.words('english')),
    'es': set(stopwords.words('spanish')),
    'de': set(stopwords.words('german')),
    'fr': set(stopwords.words('french')),
    'ru': set(stopwords.words('russian')),
}

combined_stop_words = set().union(*stop_words_multilingual.values())

# Function to remove stopwords
def remove_stopwords(word_list):
    return [word for word in word_list if word not in combined_stop_words]


def vectorize_text_columns(df, pkl_dir, vectorizers_dir):
    """
    Vectorizes text columns using TF-IDF and serializes the resulting models and matrices.
    """
    logger = logging.getLogger('preprocess_logger')

    # Apply stopword removal
    df['title_words'] = df['title_words'].apply(remove_stopwords)
    df['tags_words'] = df['tags_words'].apply(remove_stopwords)
    df['description_words'] = df['description_words'].apply(remove_stopwords)

    # Convert word lists to strings
    df['title_words_str'] = df['title_words'].apply(lambda x: ' '.join(x))
    df['tags_words_str'] = df['tags_words'].apply(lambda x: ' '.join(x))
    df['description_words_str'] = df['description_words'].apply(lambda x: ' '.join(x))

    # TF-IDF Vectorization
    title_vectorizer = TfidfVectorizer(max_features=10000)
    tags_vectorizer = TfidfVectorizer(max_features=10000)
    description_vectorizer = TfidfVectorizer(max_features=10000)

    # Fit and transform
    title_tfidf = title_vectorizer.fit_transform(df['title_words_str'])
    tags_tfidf = tags_vectorizer.fit_transform(df['tags_words_str'])
    description_tfidf = description_vectorizer.fit_transform(df['description_words_str'])

    # Save the vectorizers
    with open(os.path.join(vectorizers_dir, 'title_vectorizer.pkl'), 'wb') as f:
        pickle.dump(title_vectorizer, f)

    with open(os.path.join(vectorizers_dir, 'tags_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tags_vectorizer, f)

    with open(os.path.join(vectorizers_dir, 'description_vectorizer.pkl'), 'wb') as f:
        pickle.dump(description_vectorizer, f)

    # Save the TF-IDF matrices
    with open(os.path.join(pkl_dir, 'title_tfidf.pkl'), 'wb') as f:
        pickle.dump(title_tfidf, f)

    with open(os.path.join(pkl_dir, 'tags_tfidf.pkl'), 'wb') as f:
        pickle.dump(tags_tfidf, f)

    with open(os.path.join(pkl_dir, 'description_tfidf.pkl'), 'wb') as f:
        pickle.dump(description_tfidf, f)

    logger.info("Vectorization completed")


def analyze_and_remove_duplicates(df):
    """
    Analyzes and removes duplicates based on 'channel_title_cleaned' and 'title_words_str'.
    """
    logger = logging.getLogger('preprocess_logger')

    # Create string representation for 'title_words'
    df['title_words_str'] = df['title_words'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

    # Identify duplicates
    duplicated_videos = df[df.duplicated(subset=['channel_title_cleaned', 'title_words_str'], keep=False)]
    duplicated_videos_sorted = duplicated_videos.sort_values(by=['channel_title_cleaned', 'title_words_str']).reset_index(drop=True)
    duplicated_videos_sorted['duplicate_count'] = duplicated_videos_sorted.groupby(['channel_title_cleaned', 'title_words_str'])['title_words_str'].transform('count')

    logger.info(f"Identified {duplicated_videos_sorted['duplicate_count'].nunique()} groups of duplicates.")

    # Remove duplicates, keeping the first occurrence
    df_cleaned = df.drop_duplicates(subset=['channel_title_cleaned', 'title_words_str'], keep='first')
    df_cleaned = df_cleaned.drop(columns=['title_words_str'])

    logger.info(f"Duplicates removed. Clean DataFrame has {len(df_cleaned)} records.")

    return df_cleaned


def preprocess_data(prep_dir, output_dir, pkl_dir, vectorizers_dir):
    """
    Executes the declared functions, loads and concatenates regional files from prep_dir, 
    then applies the cleaning, imputation, and transformation process to the combined dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('preprocess_logger')

    all_data = []
    for filename in os.listdir(prep_dir):
        # Uses a regular expression to filter files like 'USvideos.csv'
        if re.match(r'^[A-Z]{2}videos\.csv$', filename):
            csv_path = os.path.join(prep_dir, filename)
            country_code = filename[:2]
            json_filename = country_code + '_category_id.json'
            json_path = os.path.join(prep_dir, json_filename)

            if not os.path.exists(json_path):
                logging.error(f"JSON file {json_filename} not found for CSV file {filename}.")
                continue

            try:
                # Load the CSV
                data = pd.read_csv(csv_path, encoding='utf-8')
                data['country'] = country_code

                # Load the JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    categories = json.load(f)['items']
                category_dict = {int(cat['id']): cat['snippet']['title'] for cat in categories}
                data['category_name'] = data['category_id'].map(category_dict)

                all_data.append(data)
                logger.info(f"File loaded: {filename}")

            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                continue
        else:
            logger.info(f"Skipping file: {filename}")

    # Concatenate and save all data in prep_dir
    if all_data:
        all_data_df = pd.concat(all_data, ignore_index=True)
        preprocessed_output_path = os.path.join(prep_dir, 'all_data_preprocessed.csv')
        all_data_df.to_csv(preprocessed_output_path, index=False)
        logger.info(f"Concatenated data saved in '{preprocessed_output_path}'")
    else:
        logging.error("No data files were loaded.")
        return

    # Concatenate and save all data in prep_dir
    all_data_df = pd.concat(all_data)
    preprocessed_output_path = os.path.join(prep_dir, 'all_data_preprocessed.csv')
    all_data_df.to_csv(preprocessed_output_path, index=False)
    logger.info(f"Concatenated data saved in '{preprocessed_output_path}'")

    # Reapply the cleaning and transformation process on the combined dataset
    df = all_data_df.copy()

    # Data cleaning
    df = clean_data(df)

    # Date transformation
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    df['publish_month'] = df['publish_time'].dt.month
    df['publish_day'] = df['publish_time'].dt.day
    df['publish_hour'] = df['publish_time'].dt.hour
    df['publish_year'] = df['publish_time'].dt.year
    df['publish_date'] = df['publish_time'].dt.date
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
    df['trending_year'] = df['trending_date'].dt.year
    df['trending_month'] = df['trending_date'].dt.month
    df['trending_day'] = df['trending_date'].dt.day
    df['trending_day_of_week'] = df['trending_date'].dt.dayofweek

    # Text column transformation
    df = transform_text_columns(df)
    
    # Feature engineering
    df = feature_engineering(df)

    # Standardization and encoding
    label_encoder = LabelEncoder()
    df['channel_title_encoded'] = label_encoder.fit_transform(df['channel_title'])

    # Standardization of numerical features
    numeric_features = df[['views', 'likes', 'dislikes', 'like_dislike_ratio', 'log_views', 
                           'publish_year', 'publish_month', 'publish_day', 'publish_hour', 
                           'days_to_trend', 'channel_title_encoded']]
    scaler = StandardScaler()
    df[numeric_features.columns] = scaler.fit_transform(numeric_features)

    # Processing of boolean columns
    binary_columns = ['comments_disabled', 'ratings_disabled', 'video_error_or_removed']
    df[binary_columns] = df[binary_columns].astype(int)

    # Duplicate analysis and removal
    df = analyze_and_remove_duplicates(df)

    # Save the processed DataFrame in output_dir
    final_output_path = os.path.join(output_dir, 'all_data_processed.csv')
    df.to_csv(final_output_path, index=False)
    logger.info(f"Processed data saved in: {final_output_path}")

    # Serialize the clean DataFrame in the 'pkl' folder
    with open(os.path.join(pkl_dir, 'all_data_processed.pkl'), 'wb') as f:
        pickle.dump(df, f)
    logger.info("Clean DataFrame serialized and saved in the 'pkl' folder")

    return df

def main():
    logger = setup_logger('preprocess_logger')
    logger.info("Data preprocessing started.")
    
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to raw data')
    parser.add_argument('--output_dir', type=str, required=True, help='Path for processed data')
    args = parser.parse_args()

    prep_dir = os.path.join(args.output_dir, 'pre_processed')
    output_dir = os.path.join(args.output_dir, 'processed')
    pkl_dir = os.path.join(output_dir, 'pkl')
    vectorizers_dir = os.path.join(output_dir, 'vectorizers')

    ensure_dir_exists(prep_dir)
    ensure_dir_exists(output_dir)
    ensure_dir_exists(pkl_dir)
    ensure_dir_exists(vectorizers_dir)

    prepare_files(args.input_dir, prep_dir)
    df_processed = preprocess_data(prep_dir, output_dir, pkl_dir, vectorizers_dir)
    vectorize_text_columns(df_processed, pkl_dir, vectorizers_dir)
    logger.info("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
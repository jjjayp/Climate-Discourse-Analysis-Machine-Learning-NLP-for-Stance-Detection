import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud

def preprocess_text(text):
    if pd.isna(text) or text is None:
        return ''
    
    text = str(text).lower()
    
    # Remove URLs, punctuation, and digits
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    
    return text

def combine_text_fields(row):
    fields = [
        str(row['post_title']) if not pd.isna(row['post_title']) else '',
        str(row['post_self_text']) if not pd.isna(row['post_self_text']) else '',
        str(row['self_text']) if not pd.isna(row['self_text']) else ''
    ]
    return ' '.join(field for field in fields if field)


def get_top_words_per_subreddit(df, n_words=50, min_df=5):
    df['combined_text'] = df.apply(combine_text_fields, axis=1)
    df['combined_text'] = df['combined_text'].apply(preprocess_text)
    df = df[df['combined_text'].str.len() > 0].copy()
    
    # Custom stop words to remove generic terms
    custom_stop_words = {
        'one', 'like', 'get', 'know', 'think', 'would', 'could', 
        'really', 'want', 'much', 'make', 'going', 'something', 
        'said', 'way', 'also', 'did', 'got', 'see', 'come'
    }
    
    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=min_df,
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    
    subreddit_top_words = {}
    
    for subreddit in df['subreddit'].unique():
        subreddit_texts = df[df['subreddit'] == subreddit]['combined_text']
        
        if len(subreddit_texts) == 0:
            continue
            
        try:
            tfidf_matrix = tfidf.fit_transform(subreddit_texts)
            feature_names = np.array(tfidf.get_feature_names_out())
            
            mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # Filter out custom stop words
            valid_indices = [
                i for i, word in enumerate(feature_names) 
                if word not in custom_stop_words
            ]
            
            feature_names = feature_names[valid_indices]
            mean_tfidf = mean_tfidf[valid_indices]
            
            top_indices = mean_tfidf.argsort()[-n_words:][::-1]
            
            subreddit_top_words[subreddit] = {
                'words': feature_names[top_indices],
                'scores': mean_tfidf[top_indices]
            }
        except ValueError as e:
            print(f"Error processing subreddit {subreddit}: {str(e)}")
            continue
    
    # Remove common words between subreddits
    subreddit_top_words = subreddit_top_words
    return subreddit_top_words

def create_word_clouds(top_words_dict):
    plt.figure(figsize=(20, 10))
    
    for idx, (subreddit, data) in enumerate(top_words_dict.items(), 1):
        # Create word frequency dictionary
        word_freq = {word: score for word, score in zip(data['words'], data['scores'])}
        
        # Create WordCloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            colormap='viridis'
        ).generate_from_frequencies(word_freq)
        
        # Plot
        plt.subplot(1, len(top_words_dict), idx)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'r/{subreddit} Distinctive Words')
        plt.axis('off')
    
    plt.tight_layout()
    return plt

def analyze_subreddit_texts(df, n_words=50):
    print("Initial data shape:", df.shape)
    print("\nMissing values in text columns:")
    print(df[['post_title', 'post_self_text', 'self_text']].isna().sum())
    
    top_words_dict = get_top_words_per_subreddit(df, n_words=n_words)
    
    print("\nMost distinctive words in each subreddit:")
    for subreddit, data in top_words_dict.items():
        print(f"\nr/{subreddit} top words:")
        for word, score in zip(data['words'], data['scores']):
            print(f"{word}: {score:.4f}")
    
    # Create and display word clouds
    fig = create_word_clouds(top_words_dict)
    
    return top_words_dict, fig

# Read the CSV file
df = pd.read_csv('project10.csv')

# Analyze subreddit texts and create word clouds
top_words_dict, fig = analyze_subreddit_texts(df)
plt.show()
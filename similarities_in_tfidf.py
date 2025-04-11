from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_vectors(df):
    """
    Build TF-IDF vectors from preprocessed text
    """
    # Combine and preprocess text
    df['combined_text'] = df.apply(combine_text_fields, axis=1)
    df['combined_text'] = df['combined_text'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['combined_text'].str.len() > 0].copy()
    
    # Initialize and fit TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=None,  # Keep all features
        stop_words='english',
        min_df=5,  # Minimum document frequency
        token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words with 3+ letters
    )
    
    # Fit and transform the text data
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    
    return tfidf, tfidf_matrix, df

def analyze_word_similarities(tfidf, tfidf_matrix, selected_words):
    """
    Compute cosine similarities between selected words
    """
    # Get feature names and their indices
    feature_names = tfidf.get_feature_names_out()
    word_indices = {}
    
    # Find indices of selected words
    for word in selected_words:
        try:
            word_indices[word] = np.where(feature_names == word)[0][0]
        except IndexError:
            print(f"Warning: Word '{word}' not found in vocabulary")
            return None
    
    # Extract TF-IDF vectors for selected words
    word_vectors = tfidf_matrix.T[list(word_indices.values())]
    
    # Compute pairwise cosine similarities
    similarities = cosine_similarity(word_vectors)
    
    return similarities, list(word_indices.keys())

def plot_similarities(similarities, words):
    """
    Create a heatmap of word similarities
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarities, 
                annot=True, 
                fmt='.3f', 
                cmap='YlOrRd',
                xticklabels=words,
                yticklabels=words)
    plt.title('Cosine Similarity Between Selected Words')
    plt.tight_layout()
    return plt.gcf()

tfidf, tfidf_matrix, processed_df = build_tfidf_vectors(df)

# Print dimensions of feature vectors
n_documents, n_features = tfidf_matrix.shape
print(f"\nFeature Vector Dimensions:")
print(f"Number of documents: {n_documents}")
print(f"Number of features (unique words): {n_features}")

# Selected prominent words from each subreddit
# Replace these with your selected words based on Task 1
selected_words = ['climate', 'change', 'action', 'global']  # Example words

# Compute similarities
similarities, found_words = analyze_word_similarities(tfidf, tfidf_matrix, selected_words)

if similarities is not None:
    # Plot similarities
    fig = plot_similarities(similarities, found_words)
    plt.show()
    
    # Print similarity matrix
    print("\nPairwise Cosine Similarities:")
    similarity_df = pd.DataFrame(similarities, 
                               index=found_words,
                               columns=found_words)
    print(similarity_df)
    
    # Print additional statistics
    print("\nMean similarity score:", np.mean(similarities[np.triu_indices_from(similarities, k=1)]))
    print("Max similarity score:", np.max(similarities[np.triu_indices_from(similarities, k=1)]))
    print("Min similarity score:", np.min(similarities[np.triu_indices_from(similarities, k=1)]))

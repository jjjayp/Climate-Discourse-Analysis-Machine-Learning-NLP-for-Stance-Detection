import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data
df = pd.read_csv('project10.csv')

def preprocess_text(text):
    if pd.isna(text) or text is None:
        return ''
    
    text = str(text).lower()
    
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
            
            top_indices = mean_tfidf.argsort()[-n_words:][::-1]
            
            subreddit_top_words[subreddit] = {
                'words': feature_names[top_indices],
                'scores': mean_tfidf[top_indices]
            }
        except ValueError as e:
            print(f"Error processing subreddit {subreddit}: {str(e)}")
            continue
    
    return subreddit_top_words

# Preprocess the text
df['combined_text'] = df.apply(combine_text_fields, axis=1)
df['combined_text'] = df['combined_text'].apply(preprocess_text)
df = df[df['combined_text'].str.len() > 0].copy()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['combined_text'], df['subreddit'], test_size=0.2, random_state=42, stratify=df['subreddit'])

# Build TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000, min_df=5, stop_words='english', token_pattern=r'\b[a-zA-Z]{3,}\b')

# Simple average of word vectors
print("Building sentence vectors using simple averaging...")
X_train_avg = np.array([np.mean([w2v_model[w] for w in text.split() if w in w2v_model], axis=0) for text in X_train])
X_test_avg = np.array([np.mean([w2v_model[w] for w in text.split() if w in w2v_model], axis=0) for text in X_test])

# TF-IDF weighted average of word vectors
print("Building sentence vectors using TF-IDF weighted averaging...")
tfidf.fit(X_train)
X_train_tfidf = np.array([np.average([w2v_model[w] for w in text.split() if w in w2v_model], axis=0, weights=[tfidf.idf_[tfidf.vocabulary_.get(w, 0)] for w in text.split() if w in w2v_model]) for text in X_train])
X_test_tfidf = np.array([np.average([w2v_model[w] for w in text.split() if w in w2v_model], axis=0, weights=[tfidf.idf_[tfidf.vocabulary_.get(w, 0)] for w in text.split() if w in w2v_model]) for text in X_test])

# Train and evaluate models
print("Training and evaluating models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': LinearSVC(max_iter=1000)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Simple average
    model.fit(X_train_avg, y_train)
    y_pred_avg = model.predict(X_test_avg)
    print(f"{name} (Simple Average):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_avg):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_avg, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_avg, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred_avg, average='weighted'):.4f}")
    
    # TF-IDF weighted average
    model.fit(X_train_tfidf, y_train)
    y_pred_tfidf = model.predict(X_test_tfidf)
    print(f"{name} (TF-IDF Weighted Average):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_tfidf):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_tfidf, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_tfidf, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred_tfidf, average='weighted'):.4f}")

    X_train_aug = np.concatenate((X_train_tfidf, df.loc[X_train.index, ['post_score', 'post_upvote_ratio', 'post_thumbs_ups']].values), axis=1)
X_test_aug = np.concatenate((X_test_tfidf, df.loc[X_test.index, ['post_score', 'post_upvote_ratio', 'post_thumbs_ups']].values), axis=1)

# Train and evaluate models
print("Training and evaluating models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': LinearSVC(max_iter=1000)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # TF-IDF weighted average with metadata
    model.fit(X_train_aug, y_train)
    y_pred_aug = model.predict(X_test_aug)
    print(f"{name} (TF-IDF Weighted Average + Metadata):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_aug):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_aug, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_aug, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred_aug, average='weighted'):.4f}")
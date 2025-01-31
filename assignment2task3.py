from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def prepare_data(df):
    """
    Prepare data for classification
    """
    # Combine and preprocess text
    df['combined_text'] = df.apply(combine_text_fields, axis=1)
    df['combined_text'] = df['combined_text'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['combined_text'].str.len() > 0].copy()
    
    return df

def build_features(train_texts, test_texts):
    """
    Build TF-IDF features for train and test sets
    """
    tfidf = TfidfVectorizer(
        max_features=5000,  # Limit features to prevent overfitting
        min_df=5,
        stop_words='english',
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    
    # Fit on training data only to prevent data leakage
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)
    
    return X_train, X_test, tfidf

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple classifiers
    """
    # Initialize classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'SVM': LinearSVC(max_iter=1000)
    }
    
    # Store results
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-score': f1_score(y_test, y_pred, average='weighted')
        }
        
        results[name] = metrics
        
        # Print detailed classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
    
    return results

def plot_results(results):
    """
    Create visualizations of model performance
    """
    # Prepare data for plotting
    metrics_df = pd.DataFrame(results).T
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Model Performance Comparison')
    plt.ylabel('Model')
    plt.xlabel('Metric')
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar', width=0.8)
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return plt.gcf()

df_processed = prepare_data(df)

# Split into train and test sets
print("\nSplitting data into train and test sets...")
train_df, test_df = train_test_split(
    df_processed,
    test_size=0.2,
    random_state=42,
    stratify=df_processed['subreddit']
)

# Build features
print("Building TF-IDF features...")
X_train, X_test, tfidf = build_features(
    train_df['combined_text'],
    test_df['combined_text']
)

# Get labels
y_train = train_df['subreddit']
y_test = test_df['subreddit']

# Print feature dimen sions
print(f"\nFeature dimensions:")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train and evaluate models
print("\nTraining and evaluating models...")
results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# Plot results
plot_results(results)
plt.show()

# Print summary
print("\nSummary of Results:")
results_df = pd.DataFrame(results).T
print(results_df.round(3))

# Find best model for each metric
print("\nBest Models:")
for metric in results_df.columns:
    best_model = results_df[metric].idxmax()
    best_score = results_df[metric].max()
    print(f"{metric}: {best_model} ({best_score:.3f})")
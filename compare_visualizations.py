from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler

def prepare_features(df, optimal_vocab_size, metadata_columns, scaler_type='standard'):
    """
    Prepare text and metadata features
    
    Parameters:
    df: DataFrame containing text and metadata
    optimal_vocab_size: Optimal vocabulary size determined from Task 4
    metadata_columns: List of metadata column names
    scaler_type: 'standard' for StandardScaler or 'minmax' for MinMaxScaler
    """
    
    # Prepare text features
    vectorizer = TfidfVectorizer(
        max_features=optimal_vocab_size,
        min_df=5,
        stop_words='english',
        token_pattern=r'\b[a-zA-Z]{3,}\b',
        binary=True
    )
    
    # Initialize scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    return vectorizer, scaler

def train_evaluate_model(clf, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a single classifier
    """
    start_time = time.time()
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'runtime': runtime
    }
    
    return metrics

def compare_models(df, optimal_vocab_size, metadata_columns, random_state=42):
    """
    Compare models with and without metadata features
    """
    # Prepare data
    X_text = df['combined_text']
    X_meta = df[metadata_columns]
    y = df['subreddit']
    
    # Split data
    X_text_train, X_text_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
        X_text, X_meta, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Prepare features
    vectorizer, scaler = prepare_features(df, optimal_vocab_size, metadata_columns)
    
    # Transform text features
    X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
    X_text_test_tfidf = vectorizer.transform(X_text_test)
    
    # Transform metadata features
    X_meta_train_scaled = scaler.fit_transform(X_meta_train)
    X_meta_test_scaled = scaler.transform(X_meta_test)
    
    # Combine features
    X_train_combined = hstack([X_text_train_tfidf, X_meta_train_scaled])
    X_test_combined = hstack([X_text_test_tfidf, X_meta_test_scaled])
    
    # Initialize classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Naive Bayes': MultinomialNB(),
        'SVM': LinearSVC(max_iter=1000)
    }

    
    # Store results
    results = {
        'text_only': {},
        'with_metadata': {}
    }
    
    # Evaluate models with text features only
    print("\nEvaluating models with text features only...")
    for name, clf in classifiers.items():
        results['text_only'][name] = train_evaluate_model(
            clf, X_text_train_tfidf, X_text_test_tfidf, y_train, y_test
        )
    
    # Evaluate models with combined features
    print("\nEvaluating models with text and metadata features...")
    for name, clf in classifiers.items():
        results['with_metadata'][name] = train_evaluate_model(
            clf, X_train_combined, X_test_combined, y_train, y_test
        )
    
    return results

def plot_comparison(results):
    """
    Create visualizations comparing model performance
    """
    # Prepare data for plotting
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(results['text_only'].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # Plot each metric
    for idx, metric in enumerate(metrics):
        text_scores = [results['text_only'][model][metric] for model in models]
        meta_scores = [results['with_metadata'][model][metric] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[idx].bar(x - width/2, text_scores, width, label='Text Only')
        axes[idx].bar(x + width/2, meta_scores, width, label='With Metadata')
        
        axes[idx].set_title(f'{metric.capitalize()} Comparison')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(models, rotation=45)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def calculate_relative_improvement(results):
    """
    Calculate relative improvement from adding metadata
    """
    improvements = {}
    
    for model in results['text_only'].keys():
        improvements[model] = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            base_score = results['text_only'][model][metric]
            meta_score = results['with_metadata'][model][metric]
            rel_improvement = ((meta_score - base_score) / base_score) * 100
            improvements[model][metric] = rel_improvement
    
    return pd.DataFrame(improvements).T

# Run the analysis
print("Starting augmented features analysis...")

# Define metadata columns from Assignment #1
metadata_columns = [
    'post_score', 'post_upvote_ratio', 'post_thumbs_ups', 
]


# Use the optimal vocabulary size from Task 4
optimal_vocab_size = 20000  # Replace with your optimal size

# Run comparison
results = compare_models(df, optimal_vocab_size, metadata_columns)

# Create and display visualizations
fig = plot_comparison(results)
plt.show()

# Print detailed results
print("\nDetailed Results:")
print("\nText Features Only:")
for model, metrics in results['text_only'].items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

print("\nWith Metadata Features:")
for model, metrics in results['with_metadata'].items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Calculate and display relative improvements
improvements_df = calculate_relative_improvement(results)
print("\nRelative Improvement (%) from Adding Metadata:")
display(improvements_df.round(2))

# Create heatmap of improvements
plt.figure(figsize=(10, 6))
sns.heatmap(improvements_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
plt.title('Relative Improvement (%) from Adding Metadata')
plt.tight_layout()
plt.show()

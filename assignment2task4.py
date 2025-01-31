import time 

# Set default matplotlib style
plt.style.use('default')

# Define analysis functions
def analyze_vocab_pruning(df, vocab_sizes, random_state=42):
    """
    Analyze the impact of vocabulary pruning on model performance and runtime
    """
    # Prepare data
    X = df['combined_text']
    y = df['subreddit']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    results = []
    
    for vocab_size in vocab_sizes:
        print(f"\nAnalyzing vocabulary size: {vocab_size}")
        
        # Initialize vectorizer with current vocab size
        vectorizer = TfidfVectorizer(
            max_features=vocab_size,
            min_df=5,
            stop_words='english',
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        # Transform data and time the process
        start_time = time.time()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Initialize and train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(X_train_tfidf, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_test_tfidf)
        
        # Calculate total runtime
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'vocab_size': vocab_size,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'runtime': total_time
        }
        
        results.append(metrics)
        
        print(f"Runtime: {total_time:.2f} seconds")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    return pd.DataFrame(results)

def plot_pruning_results(results_df):
    """
    Create visualizations for pruning analysis results
    """
    # Create figure and subplots with adjusted size
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot metrics vs vocabulary size
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']  # Default matplotlib colors
    
    for metric, color in zip(metrics, colors):
        ax1.plot(results_df['vocab_size'], results_df[metric], 
                marker='o', label=metric.capitalize(), color=color)
    
    ax1.set_xlabel('Vocabulary Size')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics vs Vocabulary Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot runtime vs vocabulary size
    ax2.plot(results_df['vocab_size'], results_df['runtime'], 
            marker='o', color='#ff7f0e')  # Orange
    ax2.set_xlabel('Vocabulary Size')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('Runtime vs Vocabulary Size')
    ax2.grid(True, alpha=0.3)
    
    # Create heatmap of metrics
    metrics_heatmap = results_df[metrics].T
    metrics_heatmap.columns = [f'Size_{size}' for size in results_df['vocab_size']]
    sns.heatmap(metrics_heatmap, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3)
    ax3.set_title('Performance Metrics Heatmap')
    
    # Calculate relative performance change
    relative_change = results_df[metrics].div(results_df[metrics].iloc[0]) * 100 - 100
    
    for metric, color in zip(metrics, colors):
        ax4.plot(results_df['vocab_size'], relative_change[metric], 
                marker='o', label=metric.capitalize(), color=color)
    
    ax4.set_xlabel('Vocabulary Size')
    ax4.set_ylabel('Relative Change (%)')
    ax4.set_title('Relative Performance Change vs Baseline')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Run the analysis
print("Starting vocabulary pruning analysis...")
vocab_sizes = [5000, 10000, 20000, 30000, 40000, 50000, 60000]
results_df = analyze_vocab_pruning(df, vocab_sizes)

# Create and display visualizations
fig = plot_pruning_results(results_df)
plt.show()

# Print detailed results
print("\nDetailed Results:")
display(results_df.round(4))

# Calculate and print optimal points
print("\nAnalysis of Optimal Points:")
max_performance = results_df.loc[results_df['accuracy'].idxmax()]
min_runtime = results_df.loc[results_df['runtime'].idxmin()]

print(f"\nBest Performance Point:")
print(f"Vocabulary Size: {max_performance['vocab_size']}")
print(f"Accuracy: {max_performance['accuracy']:.4f}")
print(f"Runtime: {max_performance['runtime']:.2f} seconds")

print(f"\nFastest Runtime Point:")
print(f"Vocabulary Size: {min_runtime['vocab_size']}")
print(f"Accuracy: {min_runtime['accuracy']:.4f}")
print(f"Runtime: {min_runtime['runtime']:.2f} seconds")

# Calculate and display relative performance changes
baseline = results_df.iloc[-1]
relative_perf = results_df.copy()
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    relative_perf[f'{metric}_change'] = (
        (results_df[metric] - baseline[metric]) / baseline[metric] * 100
    )

print("\nRelative Performance Change from Maximum Vocabulary Size:")
display(relative_perf[['vocab_size'] + 
                     [f'{m}_change' for m in ['accuracy', 'precision', 'recall', 'f1']]].round(2))
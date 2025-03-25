import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('project10.csv')

# Convert post_created_time to datetime
df['post_created_time'] = pd.to_datetime(df['post_created_time'])

# Feature Engineering
def compute_aggregated_time_features(df):
    features = {}
    
    # 1. Hour of day distribution
    hour_dist = df['post_created_time'].dt.hour.value_counts(normalize=True).sort_index()
    features['peak_hour'] = hour_dist.idxmax()
    features['hour_entropy'] = stats.entropy(hour_dist)
    
    # 2. Day of week distribution
    dow_dist = df['post_created_time'].dt.dayofweek.value_counts(normalize=True).sort_index()
    features['peak_day'] = dow_dist.idxmax()
    features['day_entropy'] = stats.entropy(dow_dist)
    
    # 3. Month distribution
    month_dist = df['post_created_time'].dt.month.value_counts(normalize=True).sort_index()
    features['peak_month'] = month_dist.idxmax()
    features['month_entropy'] = stats.entropy(month_dist)
    
    # 4. Posting regularity
    time_diffs = df['post_created_time'].sort_values().diff().dt.total_seconds() / 3600  # in hours
    features['mean_time_between_posts'] = time_diffs.mean()
    features['std_time_between_posts'] = time_diffs.std()
    
    # 5. Temporal density
    time_range = (df['post_created_time'].max() - df['post_created_time'].min()).total_seconds() / 3600 / 24  # in days
    features['posts_per_day'] = len(df) / time_range if time_range > 0 else 0
    
    # 6. Burstiness
    if len(time_diffs) > 1:
        features['burstiness'] = (time_diffs.std() - time_diffs.mean()) / (time_diffs.std() + time_diffs.mean())
    else:
        features['burstiness'] = 0
    return pd.Series(features)

# Compute features for each subreddit
aggregated_features = df.groupby('subreddit').apply(compute_aggregated_time_features).reset_index()

# Function to test statistical significance
def test_significance(feature):
    data1 = aggregated_features.loc[aggregated_features['subreddit'] == aggregated_features['subreddit'].iloc[0], feature]
    data2 = aggregated_features.loc[aggregated_features['subreddit'] == aggregated_features['subreddit'].iloc[1], feature]
    
    # Since we have only one value per subreddit, we can't perform statistical tests
    # Instead, we'll calculate the absolute difference and relative difference
    abs_diff = abs(data1.iloc[0] - data2.iloc[0])
    rel_diff = abs_diff / ((data1.iloc[0] + data2.iloc[0]) / 2) if (data1.iloc[0] + data2.iloc[0]) != 0 else 0
    
    return abs_diff, rel_diff

# Test significance for each new feature
results = []

for feature in aggregated_features.columns[2:]:  # Skip 'subreddit' and 'level_1' columns
    abs_diff, rel_diff = test_significance(feature)
    results.append({
        'Feature': feature,
        'Absolute Difference': abs_diff,
        'Relative Difference': rel_diff
    })

# Create and display results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.set_index('Feature')
print(results_df.to_string())

# Save results to CSV
results_df.to_csv('time_feature_differences.csv')
print("\nResults have been saved to 'time_feature_differences.csv'")

# Visualize features
plt.figure(figsize=(12, 8))
sns.heatmap(aggregated_features.set_index('subreddit').iloc[:, 1:], annot=True, cmap='coolwarm', center=0)
plt.title('Aggregated Time Features by Subreddit')
plt.tight_layout()
plt.savefig('time_features_heatmap.png')
plt.close()

print("\nHeatmap of aggregated time features has been saved as 'time_features_heatmap.png'.")

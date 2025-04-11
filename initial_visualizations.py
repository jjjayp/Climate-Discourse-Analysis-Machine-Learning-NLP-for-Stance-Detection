import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def calc_stats(group):
    return pd.DataFrame({
        'mean': group[features].mean(),
        'variance': group[features].var()
    })


df = pd.read_csv('project10.csv')

post_features = ['post_score', 'post_upvote_ratio', 'post_thumbs_ups', 'post_total_awards_received']
comment_features = ['score', 'controversiality', 'ups', 'downs']
user_features = ['user_awardee_karma', 'user_awarder_karma', 'user_link_karma', 'user_comment_karma', 'user_total_karma']

is_post_level = all(feature in df.columns for feature in post_features)
if is_post_level:
    features = post_features + user_features
else:
    features = comment_features + user_features

stats = df.groupby('subreddit').apply(calc_stats).reset_index()
stats = stats.pivot(index='level_1', columns='subreddit', values=['mean', 'variance'])

subreddits = stats.columns.get_level_values(1).unique()
new_order = [(stat, subreddit) for subreddit in subreddits for stat in ['mean', 'variance']]
stats = stats.reindex(columns=new_order)

stats = stats.round(2)
stats.index.name = 'Feature'
stats.columns.names = ['Statistic', 'Subreddit']

stats.to_csv('subreddit_statistics.csv')

def is_skewed(data):
    return abs(stats.skew(data)) > 1

# Function to determine skewness
def get_skewness(data):
    skewness = skew(data)
    if abs(skewness) < 0.5:
        return "approximately symmetrical"
    elif skewness < 0:
        return "left-skewed"
    else:
        return "right-skewed"

# Function to plot distributions and print comments
def plot_and_comment(feature):
    plt.figure(figsize=(12, 6))
    for subreddit in df['subreddit'].unique():
        data = df[df['subreddit'] == subreddit][feature]
        sns.histplot(data, kde=True, label=subreddit)
        
        # Calculate and print statistics
        mean = data.mean()
        median = data.median()
        skewness = get_skewness(data)
        print(f"\n{subreddit} - {feature}:")
        print(f"  Mean: {mean:.2f}")
        print(f"  Median: {median:.2f}")
        print(f"  Distribution: {skewness}")
        print(f"  Range: {data.min():.2f} to {data.max():.2f}")
        
    plt.title(f'Distribution of {feature} by Subreddit')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'{feature}_distribution.png')
    plt.close()

# Plot distributions for each feature
for feature in features:
    plot_and_comment(feature)

# Function to determine skewness
def is_skewed(data):
    return abs(data.skew()) > 1

# Function to perform statistical test
def perform_test(feature):
    subreddit1, subreddit2 = df['subreddit'].unique()
    data1 = df[df['subreddit'] == subreddit1][feature]
    data2 = df[df['subreddit'] == subreddit2][feature]
    
    # Check for skewness
    if is_skewed(data1) or is_skewed(data2):
        # If skewed, use Mann-Whitney U test
        statistic, p_value = mannwhitneyu(data1, data2)
        test_name = "Mann-Whitney U test"
    else:
        # If not skewed, use t-test
        statistic, p_value = ttest_ind(data1, data2)
        test_name = "T-test"
    
    return test_name, statistic, p_value

# Perform tests for each feature
results = []
for feature in features:
    test_name, statistic, p_value = perform_test(feature)
    significant = "Yes" if p_value < 0.05 else "No"
    results.append({
        'Feature': feature,
        'Test': test_name,
        'Statistic': statistic,
        'P-value': p_value,
        'Significant at 5% level': significant
    })

# Create and display results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.set_index('Feature')
print(results_df.to_string())

# Save results to CSV
results_df.to_csv('statistical_test_results.csv')
print("\nResults have been saved to 'statistical_test_results.csv'")

significant_features = results_df[results_df['Significant at 5% level'] == 'Yes'].index

for feature in significant_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, hue='subreddit', kde=True, element="step")
    plt.title(f'Distribution of {feature} by Subreddit')
    plt.savefig(f'{feature}_distribution.png')
    plt.close()

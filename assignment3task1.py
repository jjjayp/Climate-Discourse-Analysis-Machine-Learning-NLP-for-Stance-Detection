import gensim.downloader as api
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
# Visualize the similarities
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-trained Word2Vec model
print("Loading pre-trained Word2Vec model...")
w2v_model = api.load("word2vec-google-news-300")

# Get the most prominent words from each subreddit
subreddit_top_words = {
    'science': ['climate', 'change', 'fossil', 'people'],
    'action': ['climate', 'change', 'action', 'post']
}

# Compute pairwise cosine similarities
print("Calculating pairwise cosine similarities...")
similarities = []
for subreddit, words in subreddit_top_words.items():
    for word1 in words:
        for word2 in words:
            if word1 != word2:
                sim = cosine_similarity([w2v_model[word1]], [w2v_model[word2]])[0][0]
                similarities.append({
                    'word1': word1,
                    'word2': word2,
                    'subreddit': subreddit,
                    'similarity': sim
                })

similarity_df = pd.DataFrame(similarities)

# Pivot the DataFrame to create a matrix
print("\nPairwise Cosine Similarities:")
similarity_matrix = similarity_df.pivot_table(index='word1', columns='word2', values='similarity', aggfunc='mean')
print(similarity_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('Pairwise Cosine Similarities of Top Words')
plt.show()


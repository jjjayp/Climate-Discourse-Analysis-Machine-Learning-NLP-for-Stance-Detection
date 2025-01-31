import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, Dense, Dropout, SimpleRNN
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gensim.downloader as api

def combine_text_fields(row):
    fields = [
        str(row['post_title']) if not pd.isna(row['post_title']) else '',
        str(row['post_self_text']) if not pd.isna(row['post_self_text']) else '',
        str(row['self_text']) if not pd.isna(row['self_text']) else ''
    ]
    return ' '.join(field for field in fields if field)

df['combined_text'] = df.apply(combine_text_fields, axis=1)

# Basic text preprocessing
df['combined_text'] = df['combined_text'].apply(lambda x: ' '.join(word.lower() for word in str(x).split()))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['combined_text'], 
    df['subreddit'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['subreddit']
)

# Tokenization parameters
max_features = 20000  # Maximum number of words to keep
maxlen = 200         # Maximum length of each sequence

# Tokenize the text
print("Tokenizing text...")
tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Convert text to sequences and pad them
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', truncating='post')

# Load pre-trained word embeddings
print("Loading pre-trained Word2Vec model...")
w2v_model = api.load("word2vec-google-news-300")

# Create embedding matrix with error handling
print("Creating embedding matrix...")
embedding_dim = 300
embedding_matrix = np.zeros((max_features, embedding_dim))
words_found = 0

for word, i in tokenizer.word_index.items():
    if i >= max_features:
        continue
    try:
        embedding_matrix[i] = w2v_model[word]
        words_found += 1
    except KeyError:
        # If word not in W2V vocabulary, initialize randomly
        embedding_matrix[i] = np.random.normal(0, 0.1, embedding_dim)

print(f"Found {words_found} words in W2V vocabulary out of {min(len(tokenizer.word_index), max_features)} words")

# Define model architectures
def build_lstm_model(vocab_size, embedding_dim, embedding_matrix, maxlen):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(len(df['subreddit'].unique()), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def build_bilstm_model(vocab_size, embedding_dim, embedding_matrix, maxlen):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(len(df['subreddit'].unique()), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def build_cnn_model(vocab_size, embedding_dim, embedding_matrix, maxlen):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False),
        Conv1D(128, 5, activation='relu'),
        Conv1D(64, 3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(len(df['subreddit'].unique()), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def build_rnn_model(vocab_size, embedding_dim, embedding_matrix, maxlen):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False),
        SimpleRNN(128, return_sequences=True),
        SimpleRNN(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(len(df['subreddit'].unique()), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# Initialize models
models = {
    'LSTM': build_lstm_model(max_features, embedding_dim, embedding_matrix, maxlen),
    'Bi-LSTM': build_bilstm_model(max_features, embedding_dim, embedding_matrix, maxlen),
    'CNN': build_cnn_model(max_features, embedding_dim, embedding_matrix, maxlen),
    'RNN': build_rnn_model(max_features, embedding_dim, embedding_matrix, maxlen)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name} model...")
    history = model.fit(
        X_train_pad, 
        pd.get_dummies(y_train),
        epochs=5,  # Reduced epochs for demonstration
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Make predictions
    y_pred = model.predict(X_test_pad)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_numeric = pd.factorize(y_test)[0]
    
    # Calculate metrics
    results[name] = {
        'accuracy': accuracy_score(y_test_numeric, y_pred_classes),
        'precision': precision_score(y_test_numeric, y_pred_classes, average='weighted'),
        'recall': recall_score(y_test_numeric, y_pred_classes, average='weighted'),
        'f1': f1_score(y_test_numeric, y_pred_classes, average='weighted')
    }
    
    print(f"\n{name} Results:")
    for metric, value in results[name].items():
        print(f"{metric.capitalize()}: {value:.4f}")

# Print comparative results
print("\nComparative Results:")
metrics_df = pd.DataFrame(results).round(4)
print(metrics_df)
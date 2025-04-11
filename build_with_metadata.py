from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, Dense, Dropout, SimpleRNN, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gensim.downloader as api

metadata_features = ['post_score', 'post_upvote_ratio', 'post_thumbs_ups']
# Fill NaN values with median
for feature in metadata_features:
    df[feature] = df[feature].fillna(df[feature].median())

# Normalize metadata features
scaler = StandardScaler()
metadata_scaled = scaler.fit_transform(df[metadata_features])
metadata_df = pd.DataFrame(metadata_scaled, columns=metadata_features)

# Split into train and test sets
X_train_text, X_test_text, X_train_meta, X_test_meta, y_train, y_test = train_test_split(
    df['combined_text'],
    metadata_df,
    df['subreddit'],
    test_size=0.2,
    random_state=42,
    stratify=df['subreddit']
)

# Tokenization parameters
max_features = 20000
maxlen = 200
embedding_dim = 300

# Tokenize the text
print("Tokenizing text...")
tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_text)

# Convert text to sequences and pad them
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', truncating='post')

# Load pre-trained word embeddings and create embedding matrix
print("Loading pre-trained Word2Vec model and creating embedding matrix...")
w2v_model = api.load("word2vec-google-news-300")
embedding_matrix = np.zeros((max_features, embedding_dim))
words_found = 0

for word, i in tokenizer.word_index.items():
    if i >= max_features:
        continue
    try:
        embedding_matrix[i] = w2v_model[word]
        words_found += 1
    except KeyError:
        embedding_matrix[i] = np.random.normal(0, 0.1, embedding_dim)

print(f"Found {words_found} words in W2V vocabulary out of {min(len(tokenizer.word_index), max_features)} words")

# Define model architectures with metadata
def build_lstm_with_metadata(vocab_size, embedding_dim, embedding_matrix, maxlen, n_metadata_features):
    # Text input branch
    text_input = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)(text_input)
    lstm1 = LSTM(128, return_sequences=True)(embedding)
    lstm2 = LSTM(64)(lstm1)
    text_features = Dropout(0.3)(lstm2)
    
    # Metadata input branch
    metadata_input = Input(shape=(n_metadata_features,))
    metadata_dense = Dense(32, activation='relu')(metadata_input)
    
    # Combine branches
    combined = Concatenate()([text_features, metadata_dense])
    dense1 = Dense(64, activation='relu')(combined)
    dropout = Dropout(0.3)(dense1)
    output = Dense(len(df['subreddit'].unique()), activation='softmax')(dropout)
    
    model = Model(inputs=[text_input, metadata_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def build_bilstm_with_metadata(vocab_size, embedding_dim, embedding_matrix, maxlen, n_metadata_features):
    text_input = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)(text_input)
    bilstm1 = Bidirectional(LSTM(128, return_sequences=True))(embedding)
    bilstm2 = Bidirectional(LSTM(64))(bilstm1)
    text_features = Dropout(0.3)(bilstm2)
    
    metadata_input = Input(shape=(n_metadata_features,))
    metadata_dense = Dense(32, activation='relu')(metadata_input)
    
    combined = Concatenate()([text_features, metadata_dense])
    dense1 = Dense(64, activation='relu')(combined)
    dropout = Dropout(0.3)(dense1)
    output = Dense(len(df['subreddit'].unique()), activation='softmax')(dropout)
    
    model = Model(inputs=[text_input, metadata_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def build_cnn_with_metadata(vocab_size, embedding_dim, embedding_matrix, maxlen, n_metadata_features):
    text_input = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)(text_input)
    conv1 = Conv1D(128, 5, activation='relu')(embedding)
    conv2 = Conv1D(64, 3, activation='relu')(conv1)
    pool = GlobalMaxPooling1D()(conv2)
    text_features = Dropout(0.3)(pool)
    
    metadata_input = Input(shape=(n_metadata_features,))
    metadata_dense = Dense(32, activation='relu')(metadata_input)
    
    combined = Concatenate()([text_features, metadata_dense])
    dense1 = Dense(64, activation='relu')(combined)
    dropout = Dropout(0.3)(dense1)
    output = Dense(len(df['subreddit'].unique()), activation='softmax')(dropout)
    
    model = Model(inputs=[text_input, metadata_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def build_rnn_with_metadata(vocab_size, embedding_dim, embedding_matrix, maxlen, n_metadata_features):
    text_input = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)(text_input)
    rnn1 = SimpleRNN(128, return_sequences=True)(embedding)
    rnn2 = SimpleRNN(64)(rnn1)
    text_features = Dropout(0.3)(rnn2)
    
    metadata_input = Input(shape=(n_metadata_features,))
    metadata_dense = Dense(32, activation='relu')(metadata_input)
    
    combined = Concatenate()([text_features, metadata_dense])
    dense1 = Dense(64, activation='relu')(combined)
    dropout = Dropout(0.3)(dense1)
    output = Dense(len(df['subreddit'].unique()), activation='softmax')(dropout)
    
    model = Model(inputs=[text_input, metadata_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# Initialize models
models = {
    'LSTM': build_lstm_with_metadata(max_features, embedding_dim, embedding_matrix, maxlen, len(metadata_features)),
    'Bi-LSTM': build_bilstm_with_metadata(max_features, embedding_dim, embedding_matrix, maxlen, len(metadata_features)),
    'CNN': build_cnn_with_metadata(max_features, embedding_dim, embedding_matrix, maxlen, len(metadata_features)),
    'RNN': build_rnn_with_metadata(max_features, embedding_dim, embedding_matrix, maxlen, len(metadata_features))
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name} model with metadata...")
    history = model.fit(
        [X_train_pad, X_train_meta],
        pd.get_dummies(y_train),
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Make predictions
    y_pred = model.predict([X_test_pad, X_test_meta])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_numeric = pd.factorize(y_test)[0]
    
    # Calculate metrics
    results[name] = {
        'accuracy': accuracy_score(y_test_numeric, y_pred_classes),
        'precision': precision_score(y_test_numeric, y_pred_classes, average='weighted'),
        'recall': recall_score(y_test_numeric, y_pred_classes, average='weighted'),
        'f1': f1_score(y_test_numeric, y_pred_classes, average='weighted')
    }
    
    print(f"\n{name} Results with Metadata:")
    for metric, value in results[name].items():
        print(f"{metric.capitalize()}: {value:.4f}")

# Print comparative results
print("\nComparative Results with Metadata:")
metrics_df = pd.DataFrame(results).round(4)
print(metrics_df)

# Save results for comparison
metrics_df.to_csv('metadata_results.csv')

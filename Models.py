import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, classification_report

def testing(text, model_path='your_model.pkl', tokenizer_path='your_tokenizer.pkl', max_sequence_length=50):
    # Load the trained model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Load the tokenizer
    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    # Tokenize and pad the input sequences
    sequences = tokenizer.texts_to_sequences([text])
    X_test_padded = pad_sequences(sequences, maxlen=max_sequence_length)

    # Make predictions
    y_pred = model.predict(X_test_padded)

    if y_pred == 0:
        output = 'debit'
    else:
        output = 'kredit'
    
    return output
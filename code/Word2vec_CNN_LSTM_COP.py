import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

# Define climate-finance specific stopwords to remove in addition to regular stopwords
FINANCIAL_STOPWORDS = {
    'inc', 'corp', 'company', 'companies', 'reuters', 'news', 'press', 'release',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 
    'october', 'november', 'december', 'wall', 'street', 'journal'
}

# Keep some important function words for Word2Vec to maintain semantic structure
KEEP_WORDS = {'not', 'no', 'but', 'and', 'or', 'if', 'when', 'what', 'how', 'while', 'than'}

# Define climate-finance terms to standardize
CLIMATE_FINANCIAL_TERMS = {
    'q1': 'first_quarter',
    'q2': 'second_quarter',
    'q3': 'third_quarter',
    'q4': 'fourth_quarter',
    'esg': 'environmental_social_governance',
    'ghg': 'greenhouse_gas',
    'co2': 'carbon_dioxide',
    'carbon': 'carbon_emissions',
    'renewable': 'renewable_energy',
    'climate': 'climate_change',
    'global warming': 'climate_change',
    'green': 'green_energy',
    'sustainable': 'sustainability',
    'paris agreement': 'paris_climate_accord',
    'emissions': 'carbon_emissions',
    'environmental': 'environmental_impact',
    'solar': 'renewable_energy_solar',
    'wind': 'renewable_energy_wind',
    'net zero': 'net_zero_emissions',
    'clean energy': 'renewable_energy',
    'fossil': 'fossil_fuel',
    'coal': 'fossil_fuel_coal'
}

# Custom callback to plot training history after each epoch
class PlotLearningCallback(Callback):
    def __init__(self, model_type, text_col, label_col, layer_num, plot_dir):
        super().__init__()
        self.model_type = model_type
        self.text_col = text_col
        self.label_col = label_col
        self.layer_num = layer_num
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        # Keep track of metrics for each epoch
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
            
        # Collect metrics
        self.train_acc.append(logs.get('accuracy', logs.get('acc', 0)))
        self.val_acc.append(logs.get('val_accuracy', logs.get('val_acc', 0)))
        self.train_loss.append(logs.get('loss', 0))
        self.val_loss.append(logs.get('val_loss', 0))
        
        if epoch % 5 == 0 or epoch == self.params['epochs'] - 1:  # Plot every 5 epochs and last epoch
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            ax1.plot(self.train_acc, label='Training Accuracy', color='blue')
            ax1.plot(self.val_acc, label='Validation Accuracy', color='orange')
            ax1.set_title(f'Accuracy Curves: {self.model_type} with {self.text_col} ({self.label_col}, Layer {self.layer_num})')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Accuracy')
            ax1.legend(loc='lower right')
            ax1.set_ylim([0, 1])
            
            # Plot loss
            ax2.plot(self.train_loss, label='Training Loss', color='blue')
            ax2.plot(self.val_loss, label='Validation Loss', color='orange')
            ax2.set_title(f'Loss Curves: {self.model_type} with {self.text_col} ({self.label_col}, Layer {self.layer_num})')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss')
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            plt.savefig(f"{self.plot_dir}/{self.model_type}_{self.text_col.replace(' ', '_')}_{self.label_col}_layer_{self.layer_num}_epoch_{epoch}.png")
            plt.close()

class ClimateNewsDeepLearningPredictor:
    def __init__(self, csv_path, word2vec_path='/home/c.c24004260/models/word2vec-google-news-300.bin'):
        """
        Initialize the stock predictor for climate change news analysis with deep learning models.
        
        Args:
            csv_path: Path to the CSV file containing climate change news data
            word2vec_path: Path to the local Word2Vec model file
        """
        self.csv_path = csv_path
        self.word2vec_path = word2vec_path
        self.data = None
        self.layers = []
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english')).union(FINANCIAL_STOPWORDS) - KEEP_WORDS
        self.results = {}
        self.word2vec_model = None
        
        # Adjusted sequence length parameters based on dataset characteristics
        self.max_title_length = 20       # For titles
        self.max_fulltext_length = 500  # For full texts
        
        # Create directories for visualizations
        self.cnn_viz_dir = 'Climate_news_second_database/Word2vec_CNN_LSTM/visualizations_cnn/COP'
        self.lstm_viz_dir = 'Climate_news_second_database/Word2vec_CNN_LSTM/visualizations_lstm/COP'
        os.makedirs(self.cnn_viz_dir, exist_ok=True)
        os.makedirs(self.lstm_viz_dir, exist_ok=True)
        os.makedirs('Climate_news_second_database/Word2vec_CNN_LSTM/visualizations_summary/COP', exist_ok=True)
        
    def load_data(self):
        """Load and preprocess the CSV data containing climate change news."""
        self.data = pd.read_csv(self.csv_path)
        
        # Convert date strings to datetime objects with flexible formatting
        try:
            # Try pandas' automatic date parsing first
            self.data['Publication date'] = pd.to_datetime(self.data['Publication date'], dayfirst=True)
            self.data['Predicting date Short'] = pd.to_datetime(self.data['Predicting date Short'], dayfirst=True)
            self.data['Predicting date Long'] = pd.to_datetime(self.data['Predicting date Long'], dayfirst=True)
        except (ValueError, TypeError):
            # If automatic parsing fails, try manual approach
            print("Automatic date parsing failed. Trying manual conversion...")
            
            # Helper function to handle different date formats
            def parse_date_column(column):
                result = []
                for date_str in column:
                    try:
                        # Try to parse as YYYY-MM-DD
                        date = pd.to_datetime(date_str, format='%Y-%m-%d')
                    except ValueError:
                        try:
                            # Try to parse as DD/MM/YYYY
                            date = pd.to_datetime(date_str, format='%d/%m/%Y')
                        except ValueError:
                            # As a last resort, let pandas guess
                            date = pd.to_datetime(date_str, dayfirst=True)
                    result.append(date)
                return pd.Series(result)
            
            self.data['Publication date'] = parse_date_column(self.data['Publication date'])
            self.data['Predicting date Short'] = parse_date_column(self.data['Predicting date Short'])
            self.data['Predicting date Long'] = parse_date_column(self.data['Predicting date Long'])
        
        # Sort by publication date
        self.data = self.data.sort_values('Publication date')
        
        print(f"Loaded {len(self.data)} climate change news articles spanning from "
              f"{self.data['Publication date'].min().strftime('%d/%m/%Y')} "
              f"to {self.data['Publication date'].max().strftime('%d/%m/%Y')}")
        print(f"Class distribution for short-term prediction: {self.data['S_label'].value_counts().to_dict()}")
        print(f"Class distribution for long-term prediction: {self.data['L_label'].value_counts().to_dict()}")
        
        # Load pre-trained Word2Vec model from local file
        self.load_word2vec_model()
            
        return self
    
    def load_word2vec_model(self):
        """Load the Word2Vec model from a local file or download it if not available."""
        print("Loading Word2Vec model...")
        try:
            # Try to load from specified local file
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
                self.word2vec_path, binary=True)
            print(f"Loaded Word2Vec model from {self.word2vec_path}")
        except FileNotFoundError:
            print(f"Local model file not found at {self.word2vec_path}.")
            print("Checking for model in standard location...")
            
            # Check if model exists in the standard location
            standard_path = '/home/c.c24004260/models/word2vec-google-news-300.bin'
            try:
                self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
                    standard_path, binary=True)
                print(f"Loaded Word2Vec model from {standard_path}")
            except FileNotFoundError:
                # As a fallback, try to download the model
                print("Model not found locally. Please run the download_word2vec.py script first.")
                print("Attempting to download the model (this will take a while)...")
                
                try:
                    import gensim.downloader as gensim_downloader
                    self.word2vec_model = gensim_downloader.load('word2vec-google-news-300')
                    print("Downloaded and loaded Word2Vec model.")
                    
                    # Save the model for future use
                    save_dir = "models"
                    os.makedirs(save_dir, exist_ok=True)
                    saved_path = os.path.join(save_dir, "word2vec-google-news-300.bin")
                    self.word2vec_model.save_word2vec_format(saved_path, binary=True)
                    print(f"Saved model to {saved_path} for future use.")
                except Exception as e:
                    print(f"Error downloading Word2Vec model: {str(e)}")
                    print("Please run the download_word2vec.py script separately.")
                    raise
    
    def preprocess_text_for_word2vec(self, text):
        """
        Preprocess text specifically for Word2Vec embeddings.
        This preprocessing preserves more semantic structure than TF-IDF preprocessing.
        
        Args:
            text: The text to preprocess
            
        Returns:
            List of preprocessed tokens
        """
        if pd.isna(text):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Standardize climate-finance terms
        for term, replacement in CLIMATE_FINANCIAL_TERMS.items():
            text = re.sub(r'\b' + term + r'\b', replacement, text)
        
        # Standardize numerical representations with units
        text = re.sub(r'\$(\d+)([kmbt])', lambda m: m.group(1) + '_' + 
                      {'k': 'thousand', 'm': 'million', 'b': 'billion', 't': 'trillion'}[m.group(2).lower()], text)
        
        # Remove most punctuation but keep sentence structure
        text = re.sub(r'[^\w\s$%.]', ' ', text)
        
        # Tokenize text into sentences
        sentences = sent_tokenize(text)
        
        # Process each sentence
        all_tokens = []
        for sentence in sentences:
            # Tokenize into words
            tokens = word_tokenize(sentence)
            
            # Selective lemmatization and stopword removal 
            tokens = [self.lemmatizer.lemmatize(token) if token not in KEEP_WORDS else token 
                     for token in tokens if token not in self.stopwords]
            
            all_tokens.extend(tokens)
            
        return all_tokens
    
    def text_to_sequence(self, tokens, max_length):
        """
        Convert tokens to a sequence of word vectors for neural network input.
        This creates a fixed-size sequence of word vectors by padding or truncating.
        
        Args:
            tokens: List of tokens
            max_length: Maximum sequence length
            
        Returns:
            2D array of shape (max_length, 300) with word vectors
        """
        # Truncate the sequence if needed
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Initialize a sequence matrix of zeros
        sequence = np.zeros((max_length, 300))
        
        # Fill the sequence with word vectors where available
        for i, token in enumerate(tokens):
            try:
                sequence[i] = self.word2vec_model[token]
            except KeyError:
                # Word not in vocabulary, leave as zeros
                continue
                
        return sequence
    
    def define_time_windows(self):
        """Define the time windows for the sliding window approach."""
        # First layer: train (2019-2021), validation (2021-2021), test (2022-2022)
        layer1 = {
            'train_start': pd.Timestamp('2019-01-01'),
            'train_end': pd.Timestamp('2021-05-31'),
            'val_start': pd.Timestamp('2021-06-01'),
            'val_end': pd.Timestamp('2021-12-31'),
            'test_start': pd.Timestamp('2022-01-01'),
            'test_end': pd.Timestamp('2022-05-31')
        }
        
        # Second layer: train (2019-2021), validation (2022-2022), test (2022-2022)
        layer2 = {
            'train_start': pd.Timestamp('2019-01-01'),
            'train_end': pd.Timestamp('2021-12-31'),
            'val_start': pd.Timestamp('2022-01-01'),
            'val_end': pd.Timestamp('2022-05-31'),
            'test_start': pd.Timestamp('2022-06-01'),
            'test_end': pd.Timestamp('2022-12-31')
        }
        
        # Third layer: train (2019-2022), validation (2022-2022), test (2023-2023)
        layer3 = {
            'train_start': pd.Timestamp('2019-01-01'),
            'train_end': pd.Timestamp('2022-05-31'),
            'val_start': pd.Timestamp('2022-06-01'),
            'val_end': pd.Timestamp('2022-12-31'),
            'test_start': pd.Timestamp('2023-01-01'),
            'test_end': pd.Timestamp('2023-05-31')
        }
        
        self.layers = [layer1, layer2, layer3]
        return self
    
    def split_data(self, layer, text_col, label_col, model_type):
        """
        Split the data into training, validation, and test sets based on the defined time windows.
        Process text using Word2Vec embeddings and create fixed-length sequences.
        
        Args:
            layer: The time window layer
            text_col: The column containing the text data ('Title' or 'Full text')
            label_col: The column containing the labels ('S_label' or 'L_label')
            model_type: The model type ('CNN' or 'LSTM')
            
        Returns:
            train_data, val_data, test_data
        """
        train_mask = (self.data['Publication date'] >= layer['train_start']) & (self.data['Publication date'] <= layer['train_end'])
        val_mask = (self.data['Publication date'] > layer['val_start']) & (self.data['Publication date'] <= layer['val_end'])
        test_mask = (self.data['Publication date'] > layer['test_start']) & (self.data['Publication date'] <= layer['test_end'])
        
        train_data = self.data[train_mask]
        val_data = self.data[val_mask]
        test_data = self.data[test_mask]
        
        print(f"Training data: {len(train_data)} samples")
        print(f"Validation data: {len(val_data)} samples")
        print(f"Test data: {len(test_data)} samples")
        
        # For CNN/LSTM models, we create sequences of word embeddings
        if model_type == 'CNN' or model_type == 'LSTM':
            # Determine maximum sequence length based on text type
            max_length = self.max_title_length if text_col == 'Title' else self.max_fulltext_length
            
            print(f"Processing {text_col} with max sequence length: {max_length}")
            
            # Preprocess and tokenize text
            X_train_tokens = [self.preprocess_text_for_word2vec(text) for text in train_data[text_col]]
            X_val_tokens = [self.preprocess_text_for_word2vec(text) for text in val_data[text_col]]
            X_test_tokens = [self.preprocess_text_for_word2vec(text) for text in test_data[text_col]]
            
            # Analysis of token lengths
            train_token_lengths = [len(tokens) for tokens in X_train_tokens]
            print(f"Token length stats for {text_col} (train set):")
            print(f"  Min: {min(train_token_lengths) if train_token_lengths else 0}")
            print(f"  Max: {max(train_token_lengths) if train_token_lengths else 0}")
            print(f"  Mean: {np.mean(train_token_lengths) if train_token_lengths else 0:.1f}")
            print(f"  Median: {np.median(train_token_lengths) if train_token_lengths else 0:.1f}")
            
            # Convert tokens to sequences of embeddings
            X_train = np.array([self.text_to_sequence(tokens, max_length) for tokens in X_train_tokens])
            X_val = np.array([self.text_to_sequence(tokens, max_length) for tokens in X_val_tokens])
            X_test = np.array([self.text_to_sequence(tokens, max_length) for tokens in X_test_tokens])
            
            print(f"Sequence shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Get labels
        y_train = train_data[label_col].values
        y_val = val_data[label_col].values
        y_test = test_data[label_col].values
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), train_data, val_data, test_data
    
    def create_cnn_model(self, text_col):
        """
        Create a CNN model for text classification with Word2Vec embeddings.
        Different architectures for title vs full text.
        
        Args:
            text_col: The text column ('Title' or 'Full text')
        """
        # Determine input shape based on text type
        max_length = self.max_title_length if text_col == 'Title' else self.max_fulltext_length
        input_shape = (max_length, 300)
        
        # Create CNN model - different architectures based on text type
        model = Sequential()
        
        if text_col == 'Title':
            # Simpler model for titles
            # Convolutional layers
            model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))  # Reduced dropout to prevent underfitting
            
            # Flatten and dense layers
            model.add(GlobalMaxPooling1D())
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.5))  # Reduced dropout
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.0001),  # Slightly higher learning rate for simpler model
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            # More complex model for full text - reducing complexity to avoid overfitting
            # Based on learning curves, we need to fight overfitting
            model.add(Conv1D(filters=128, kernel_size=7, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.5))  # Increased dropout to combat overfitting
            
            # Flatten and dense layers
            model.add(GlobalMaxPooling1D())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model with L2 regularization
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
        return model
    
    def create_lstm_model(self, text_col):
        """
        Create an LSTM model for text classification with Word2Vec embeddings.
        Different architectures for title vs full text.
        
        Args:
            text_col: The text column ('Title' or 'Full text')
        """
        # Determine input shape based on text type
        max_length = self.max_title_length if text_col == 'Title' else self.max_fulltext_length
        input_shape = (max_length, 300)
        
        # Create LSTM model - different architectures based on text type
        model = Sequential()
        
        if text_col == 'Title':
            # Simpler model for titles
            # LSTM layer
            model.add(LSTM(32, input_shape=input_shape, dropout=0.3, recurrent_dropout=0.3))
            
            # Dense layers
            model.add(Dense(16, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.0001),  # Slightly higher learning rate for simpler model
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            # Simpler model for full text to reduce overfitting
            # Looking at the learning curves, we need to simplify the model
            model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3, input_shape=input_shape))
            
            # Dense layers - simplified
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model with reduced learning rate
            model.compile(
                optimizer=Adam(learning_rate=0.00005),  # Reduced learning rate
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
        return model
    
    def plot_final_learning_curves(self, history, model_type, text_col, label_col, layer_num, visualization_dir):
        """
        Plot final learning curves after training is complete.
        
        Args:
            history: Training history
            model_type: Model type (CNN or LSTM)
            text_col: Text column used
            label_col: Label column used
            layer_num: Layer number
            visualization_dir: Directory to save visualizations
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Get history dictionary safely
        history_dict = {}
        if hasattr(history, 'history'):
            history_dict = history.history
        
        # Determine the correct metric names
        acc_metric = 'accuracy' if 'accuracy' in history_dict else 'acc'
        val_acc_metric = 'val_accuracy' if 'val_accuracy' in history_dict else 'val_acc'
        
        # Plot accuracy
        if acc_metric in history_dict and val_acc_metric in history_dict:
            ax1.plot(history_dict[acc_metric], label='Training Accuracy', color='blue')
            ax1.plot(history_dict[val_acc_metric], label='Validation Accuracy', color='orange')
        ax1.set_title(f'Final Accuracy Curves: {model_type} with {text_col} ({label_col}, Layer {layer_num})')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='lower right')
        ax1.set_ylim([0, 1])
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot loss
        if 'loss' in history_dict and 'val_loss' in history_dict:
            ax2.plot(history_dict['loss'], label='Training Loss', color='blue')
            ax2.plot(history_dict['val_loss'], label='Validation Loss', color='orange')
        ax2.set_title(f'Final Loss Curves: {model_type} with {text_col} ({label_col}, Layer {layer_num})')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        filename = f"final_{model_type.lower()}_{text_col.replace(' ', '_')}_{label_col}_layer_{layer_num}.png"
        plt.savefig(f"{visualization_dir}/{filename}")
        print(f"Saved learning curves: {filename}")
        plt.close()
    
    def train_and_evaluate_deep_model(self, text_col, label_col, model_type):
        """
        Train and evaluate a deep learning model for a specific text column and label column.
        
        Args:
            text_col: The text column to use ('Title' or 'Full text')
            label_col: The label column to use ('S_label' or 'L_label')
            model_type: The model type to use ('CNN' or 'LSTM')
        """
        # Store results
        combination_key = f"{model_type}|{text_col}|{label_col}"
        self.results[combination_key] = {
            'accuracy': [],
            'auc': [],
            'layer_results': []
        }
        
        print(f"\n{'='*80}")
        print(f"Training {model_type} model for {text_col} and {label_col}")
        print(f"{'='*80}")
        
        visualization_dir = self.cnn_viz_dir if model_type == 'CNN' else self.lstm_viz_dir
        
        for i, layer in enumerate(self.layers):
            print(f"\nLayer {i+1}:")
            print(f"Training period: {layer['train_start'].strftime('%d/%m/%Y')} - {layer['train_end'].strftime('%d/%m/%Y')}")
            print(f"Validation period: {layer['val_start'].strftime('%d/%m/%Y')} - {layer['val_end'].strftime('%d/%m/%Y')}")
            print(f"Testing period: {layer['test_start'].strftime('%d/%m/%Y')} - {layer['test_end'].strftime('%d/%m/%Y')}")
            
            # Split data
            (X_train, y_train), (X_val, y_val), (X_test, y_test), train_data, val_data, test_data = self.split_data(
                layer, text_col, label_col, model_type)
            
            # Check if there are enough samples and classes
            if len(X_train) < 10 or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
                print(f"Skipping layer {i+1} due to insufficient data or class imbalance")
                continue
            
            # Create model
            if model_type == 'CNN':
                model = self.create_cnn_model(text_col)
            else:  # 'LSTM'
                model = self.create_lstm_model(text_col)
                
            print(f"Created {model_type} model for {text_col} ({label_col})")
            model.summary()
            
            # Setup callbacks
            plot_callback = PlotLearningCallback(model_type, text_col, label_col, i+1, visualization_dir)
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=7,  # Increased patience
                restore_best_weights=True,
                verbose=1
            )
            model_checkpoint = ModelCheckpoint(
                filepath=f"{visualization_dir}/best_{model_type.lower()}_{text_col.lower().replace(' ', '_')}_{label_col}_layer_{i+1}.h5",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            
            # Train model
            print(f"Training {model_type} model...")
            # Adjust batch size based on text length - smaller batch for full text to conserve memory
            batch_size = 32 if text_col == 'Title' else 16  
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,  # Reduced max epochs
                batch_size=batch_size,
                callbacks=[early_stopping, model_checkpoint, plot_callback],
                verbose=1
            )
            
            # Evaluate on test set
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            self.results[combination_key]['accuracy'].append(accuracy)
            self.results[combination_key]['auc'].append(auc)
            
            print(f"Test Accuracy for Layer {i+1}: {accuracy:.4f}")
            print(f"Test AUC for Layer {i+1}: {auc:.4f}")
            
            # Store layer results for visualization
            layer_result = {
                'layer': i+1,
                'accuracy': accuracy,
                'auc': auc,
                'history': history.history
            }
            
            self.results[combination_key]['layer_results'].append(layer_result)
            
            # Create final learning curve visualization
            self.plot_final_learning_curves(history, model_type, text_col, label_col, i+1, visualization_dir)
        
        # Calculate average metrics
        avg_accuracy = np.mean(self.results[combination_key]['accuracy']) if self.results[combination_key]['accuracy'] else 0
        avg_auc = np.mean(self.results[combination_key]['auc']) if self.results[combination_key]['auc'] else 0
        
        print(f"\nAverage Test Accuracy across all layers: {avg_accuracy:.4f}")
        print(f"Average Test AUC across all layers: {avg_auc:.4f}")
        
        self.results[combination_key]['avg_accuracy'] = avg_accuracy
        self.results[combination_key]['avg_auc'] = avg_auc
        
        return self
    
    def run_all_combinations(self):
        """Run the analysis for all combinations of text inputs, label columns, and model types."""
        # Define all combinations
        text_cols = ['Title', 'Full text']
        label_cols = ['S_label', 'L_label']
        model_types = ['CNN', 'LSTM']
        
        # Run analysis for each combination
        for model_type in model_types:
            for text_col in text_cols:
                for label_col in label_cols:
                    self.train_and_evaluate_deep_model(text_col, label_col, model_type)
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY OF RESULTS (COP)")
        print("="*80)
        
        # Create a summary table for easier comparison
        summary_data = []
        
        for combination, results in self.results.items():
            model_type, text_col, label_col = combination.split('|')
            avg_accuracy = results.get('avg_accuracy', 0)
            avg_auc = results.get('avg_auc', 0)
            
            print(f"\nCombination: {model_type} with {text_col} + {label_col}")
            print(f"Average Accuracy: {avg_accuracy:.4f}")
            print(f"Average AUC: {avg_auc:.4f}")
            
            # Print layer-specific results
            for i, accuracy in enumerate(results.get('accuracy', [])):
                auc = results.get('auc', [])[i]
                print(f"  Layer {i+1} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
            summary_data.append({
                'Combination': f"{model_type} with {text_col} + {label_col}",
                'Avg Accuracy': avg_accuracy,
                'Avg AUC': avg_auc,
                'Model': model_type,
                'Text': text_col,
                'Label': label_col
            })
        
        # Create summary visualizations
        self.create_summary_visualization(summary_data)
        
        return self
    
    def create_summary_visualization(self, summary_data):
        """Create a summary visualization comparing all model combinations."""
        if not summary_data:
            print("No data available for summary visualization")
            return
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame(summary_data)
        
        # Convert metrics to float for plotting
        df['Avg Accuracy'] = df['Avg Accuracy'].astype(float)
        df['Avg AUC'] = df['Avg AUC'].astype(float)
        
        # Split by model type
        cnn_data = df[df['Model'] == 'CNN']
        lstm_data = df[df['Model'] == 'LSTM']
        
        # 1. Performance comparison for CNN
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy and AUC bars for CNN
        x = np.arange(len(cnn_data))
        width = 0.35
        
        plt.bar(x - width/2, cnn_data['Avg Accuracy'], width, label='Average Accuracy', color='skyblue')
        plt.bar(x + width/2, cnn_data['Avg AUC'], width, label='Average AUC', color='salmon')
        
        plt.xlabel('CNN Model Combination')
        plt.ylabel('Score')
        plt.title('Performance Comparison of CNN Models')
        labels = [f"{row['Text']} + {row['Label']}" for _, row in cnn_data.iterrows()]
        plt.xticks(x, labels, rotation=0, ha='right')
        plt.legend()
        
        # Add value labels on top of bars
        for i, v in enumerate(cnn_data['Avg Accuracy']):
            plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
        
        for i, v in enumerate(cnn_data['Avg AUC']):
            plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
        
        plt.ylim(0, max(cnn_data['Avg Accuracy'].max(), cnn_data['Avg AUC'].max()) + 0.1)
        plt.tight_layout()
        
        # Save the visualization
        save_path = os.path.join('Climate_news_second_database/Word2vec_CNN_LSTM/visualizations_summary/COP', "cnn_performance_comparison.png")
        plt.savefig(save_path)
        print(f"CNN summary comparison visualization saved as: {save_path}")
        plt.close()
        
        # 2. Performance comparison for LSTM
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy and AUC bars for LSTM
        x = np.arange(len(lstm_data))
        
        plt.bar(x - width/2, lstm_data['Avg Accuracy'], width, label='Average Accuracy', color='skyblue')
        plt.bar(x + width/2, lstm_data['Avg AUC'], width, label='Average AUC', color='salmon')
        
        plt.xlabel('LSTM Model Combination')
        plt.ylabel('Score')
        plt.title('Performance Comparison of LSTM Models')
        labels = [f"{row['Text']} + {row['Label']}" for _, row in lstm_data.iterrows()]
        plt.xticks(x, labels, rotation=0, ha='right')
        plt.legend()
        
        # Add value labels on top of bars
        for i, v in enumerate(lstm_data['Avg Accuracy']):
            plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
        
        for i, v in enumerate(lstm_data['Avg AUC']):
            plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
        
        plt.ylim(0, max(lstm_data['Avg Accuracy'].max(), lstm_data['Avg AUC'].max()) + 0.1)
        plt.tight_layout()
        
        # Save the visualization
        save_path = os.path.join('Climate_news_second_database/Word2vec_CNN_LSTM/visualizations_summary/COP', "lstm_performance_comparison.png")
        plt.savefig(save_path)
        print(f"LSTM summary comparison visualization saved as: {save_path}")
        plt.close()
        
        # 3. Layer-specific performance visualizations
        self.create_layer_performance_visualizations()
    
    def create_layer_performance_visualizations(self):
        """Create visualizations showing performance across different layers for CNN and LSTM models."""
        # Prepare data for visualization
        cnn_layer_data = []
        lstm_layer_data = []
        
        for combination, results in self.results.items():
            model_type, text_col, label_col = combination.split('|')
            
            for i, accuracy in enumerate(results.get('accuracy', [])):
                auc = results.get('auc', [])[i]
                layer_info = {
                    'Model': model_type,
                    'Text': text_col,
                    'Label': label_col,
                    'Layer': f"Layer {i+1}",
                    'Layer_num': i+1,
                    'Accuracy': accuracy,
                    'AUC': auc,
                    'Combination': f"{text_col} + {label_col}"
                }
                
                if model_type == 'CNN':
                    cnn_layer_data.append(layer_info)
                else:  # LSTM
                    lstm_layer_data.append(layer_info)
        
        # Create visualizations for each model type
        self._create_model_layer_visualization(cnn_layer_data, 'CNN')
        self._create_model_layer_visualization(lstm_layer_data, 'LSTM')
    
    def _create_model_layer_visualization(self, layer_data, model_type):
        """Create layer-specific visualizations for a given model type."""
        if not layer_data:
            print(f"No layer data available for {model_type}")
            return
        
        # Create DataFrame
        df = pd.DataFrame(layer_data)
        
        # Create visualization - Accuracy by layer
        plt.figure(figsize=(14, 10))
        
        # 1. Accuracy by combination and layer
        plt.subplot(2, 1, 1)
        sns.barplot(x='Combination', y='Accuracy', hue='Layer', data=df)
        plt.title(f'{model_type} Accuracy by Model Combination and Layer')
        plt.ylim(0, max(0.8, df['Accuracy'].max() + 0.05))
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        # 2. AUC by combination and layer
        plt.subplot(2, 1, 2)
        sns.barplot(x='Combination', y='AUC', hue='Layer', data=df)
        plt.title(f'{model_type} AUC by Model Combination and Layer')
        plt.ylim(0, max(0.8, df['AUC'].max() + 0.05))
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        
        # Save the visualization
        save_path = os.path.join('Climate_news_second_database/Word2vec_CNN_LSTM/visualizations_summary/COP', f"{model_type.lower()}_layer_performance.png")
        plt.savefig(save_path)
        print(f"{model_type} layer performance visualization saved as: {save_path}")
        plt.close()


# Main execution
if __name__ == "__main__":
    # Ensure all visualization directories exist
    for directory in ['Climate_news_second_database/Word2vec_CNN_LSTM/visualizations_cnn/COP', 'Climate_news_second_database/Word2vec_CNN_LSTM/visualizations_lstm/COP', 'Climate_news_second_database/Word2vec_CNN_LSTM/visualizations_summary/COP', 'models']:
        os.makedirs(directory, exist_ok=True)
    
    # Customize paths as needed
    csv_path = '/home/c.c24004260/Climate_news_second_database/us_news_semantics_COP_completed.csv'
    word2vec_path = '/home/c.c24004260/models/word2vec-google-news-300.bin'
    
    predictor = ClimateNewsDeepLearningPredictor(csv_path, word2vec_path)
    predictor.load_data().define_time_windows().run_all_combinations()
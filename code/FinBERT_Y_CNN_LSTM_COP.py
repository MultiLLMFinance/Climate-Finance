import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os
import nltk
import torch
import gc
from transformers import AutoTokenizer, AutoModel
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM, GlobalMaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow to use CPU, but allow PyTorch to use GPU
# Only hide GPUs from TensorFlow
tf.config.set_visible_devices([], 'GPU')
print("TensorFlow configured to use CPU only, PyTorch will use GPU if available")

# Download required NLTK resources for basic preprocessing
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

# Define climate-finance specific terms to standardize
CLIMATE_FINANCIAL_TERMS = {
    'q1': 'first quarter',
    'q2': 'second quarter',
    'q3': 'third quarter',
    'q4': 'fourth quarter',
    'esg': 'environmental social governance',
    'ghg': 'greenhouse gas',
    'co2': 'carbon dioxide',
    'carbon': 'carbon emissions',
    'renewable': 'renewable energy',
    'climate': 'climate change',
    'global warming': 'climate change',
    'green': 'green energy',
    'sustainable': 'sustainability',
    'paris agreement': 'paris climate accord',
    'emissions': 'carbon emissions',
    'environmental': 'environmental impact',
    'solar': 'renewable energy solar',
    'wind': 'renewable energy wind',
    'net zero': 'net zero emissions',
    'clean energy': 'renewable energy',
    'fossil': 'fossil fuel',
    'coal': 'fossil fuel coal'
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
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot loss
            ax2.plot(self.train_loss, label='Training Loss', color='blue')
            ax2.plot(self.val_loss, label='Validation Loss', color='orange')
            ax2.set_title(f'Loss Curves: {self.model_type} with {self.text_col} ({self.label_col}, Layer {self.layer_num})')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss')
            ax2.legend(loc='upper right')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(f"{self.plot_dir}/{self.model_type}_{self.text_col.replace(' ', '_')}_{self.label_col}_layer_{self.layer_num}_epoch_{epoch}.png")
            plt.close()

class ClimateNewsFinBERTPredictor:
    def __init__(self, csv_path, finbert_model="yiyanghkust/finbert-esg-9-categories", hf_token=None):
        """
        Initialize the stock predictor for climate change news analysis using FinBERT embeddings.
        
        Args:
            csv_path: Path to the CSV file containing climate change news data
            finbert_model: The FinBERT model to use (default: "yiyanghkust/finbert-esg-9-categories")
            hf_token: Optional Hugging Face token for API authentication
        """
        self.csv_path = csv_path
        self.finbert_model_name = finbert_model
        self.hf_token = hf_token
        self.data = None
        self.layers = []
        self.results = {}
        self.tokenizer = None
        self.finbert_model = None
        
        # Set max sequence lengths for different text types
        self.max_title_length = 20      # For titles 
        self.max_fulltext_length = 500  # For full texts 
        
        # Create directories for visualizations
        self.cnn_viz_dir = 'Climate_news_second_database/FinBERT_Y_CNN_LSTM/visualizations_cnn/COP'
        self.lstm_viz_dir = 'Climate_news_second_database/FinBERT_Y_CNN_LSTM/visualizations_lstm/COP'
        os.makedirs(self.cnn_viz_dir, exist_ok=True)
        os.makedirs(self.lstm_viz_dir, exist_ok=True)
        os.makedirs('Climate_news_second_database/FinBERT_Y_CNN_LSTM/visualizations_summary/COP', exist_ok=True)
        
        # Load FinBERT model and tokenizer
        self.load_finbert_model()
            
    def load_data(self):
        """Load and preprocess the CSV data containing climate change news."""
        print("Loading data...")
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
        
        return self
    
    def load_finbert_model(self):
        """Load the FinBERT model and tokenizer."""
        print(f"Loading FinBERT model: {self.finbert_model_name}")
        
        # Explicitly manage cache directories
        cache_dir = "/scratch/c.c24004260/model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set environment variables
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        
        # Use token if provided
        if self.hf_token:
            print("Using provided Hugging Face token for authentication")
            self.tokenizer = AutoTokenizer.from_pretrained(self.finbert_model_name, token=self.hf_token,cache_dir=cache_dir)
            self.finbert_model = AutoModel.from_pretrained(self.finbert_model_name, token=self.hf_token,cache_dir=cache_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.finbert_model_name,cache_dir=cache_dir)
            self.finbert_model = AutoModel.from_pretrained(self.finbert_model_name,cache_dir=cache_dir)
            
        # Check if GPU is available and move model to GPU if possible
        if torch.cuda.is_available():
            self.finbert_model = self.finbert_model.to('cuda')
            print("FinBERT model loaded on GPU")
        else:
            print("GPU not available, running on CPU")
            
        print("FinBERT model and tokenizer loaded successfully")
        return self
    
    def preprocess_text_for_finbert(self, text):
        """
        Preprocess text for FinBERT.
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text ready for FinBERT tokenizer
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase (FinBERT was trained on lowercase text)
        text = text.lower()
        
        # Standardize climate-finance terms
        for term, replacement in CLIMATE_FINANCIAL_TERMS.items():
            text = re.sub(r'\b' + term + r'\b', replacement, text)
        
        # Standardize numerical representations with units
        text = re.sub(r'\$(\d+)([kmbt])', lambda m: m.group(1) + ' ' + 
                      {'k': 'thousand', 'm': 'million', 'b': 'billion', 't': 'trillion'}[m.group(2).lower()], text)
        
        # Clean text - remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_finbert_embeddings(self, texts, max_length):
        """Extract FinBERT embeddings with batch processing to reduce memory usage."""
        # For full text, use smaller batches to avoid OOM
        batch_size = 50 if max_length > 100 else 64  # Smaller batches for longer texts
        
        all_embeddings = []
        all_masks = []
    
        for i in range(0, len(texts), batch_size):
            # Process a single batch
            batch_texts = texts[i:i+batch_size]
            processed_texts = [self.preprocess_text_for_finbert(text) for text in batch_texts]
        
            # Tokenize batch
            encoded_texts = self.tokenizer(
                processed_texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Extract embeddings for batch
            with torch.no_grad():
                device = self.finbert_model.device
                input_ids = encoded_texts['input_ids'].to(device)
                attention_mask = encoded_texts['attention_mask'].to(device)
                
                outputs = self.finbert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                last_hidden_states = outputs.last_hidden_state
                batch_embeddings = last_hidden_states.cpu().numpy()
                batch_masks = encoded_texts['attention_mask'].numpy()
                
            all_embeddings.append(batch_embeddings)
            all_masks.append(batch_masks)
        
            # Explicitly clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log progress for long sequences
            if len(texts) > 100 and i % (batch_size * 5) == 0:
                print(f"  Processed {i}/{len(texts)} samples...")
    
        # Combine all batches
        return np.vstack(all_embeddings), np.vstack(all_masks)
    
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
        Extract FinBERT embeddings for text data.
        
        Args:
            layer: The time window layer
            text_col: The column containing the text data ('Title' or 'Full text')
            label_col: The column containing the labels ('S_label' or 'L_label')
            model_type: The model type ('CNN' or 'LSTM')
            
        Returns:
            train_data, val_data, test_data with embeddings
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
        
        # Determine maximum sequence length based on text type
        max_length = self.max_title_length if text_col == 'Title' else self.max_fulltext_length
        
        print(f"Processing {text_col} with max sequence length: {max_length}")
        
        # Extract FinBERT embeddings
        print("Extracting FinBERT embeddings for training data...")
        X_train_embeddings, X_train_mask = self.get_finbert_embeddings(train_data[text_col].tolist(), max_length)
        
        print("Extracting FinBERT embeddings for validation data...")
        X_val_embeddings, X_val_mask = self.get_finbert_embeddings(val_data[text_col].tolist(), max_length)
        
        print("Extracting FinBERT embeddings for test data...")
        X_test_embeddings, X_test_mask = self.get_finbert_embeddings(test_data[text_col].tolist(), max_length)
        
        # Get labels
        y_train = train_data[label_col].values
        y_val = val_data[label_col].values
        y_test = test_data[label_col].values
        
        print(f"Embeddings shapes - Train: {X_train_embeddings.shape}, Val: {X_val_embeddings.shape}, Test: {X_test_embeddings.shape}")
        
        return (X_train_embeddings, X_train_mask, y_train), (X_val_embeddings, X_val_mask, y_val), (X_test_embeddings, X_test_mask, y_test), train_data, val_data, test_data
    
    def create_cnn_model(self, text_col, input_shape):
        """
        Create a CNN model for text classification with FinBERT embeddings.
        Different architectures for title vs full text.
        
        Args:
            text_col: The text column ('Title' or 'Full text')
            input_shape: Shape of the input data
            
        Returns:
            Compiled CNN model
        """
        """Create a CPU-only CNN model."""
        print("Creating CNN model on CPU")
        
        inputs = Input(shape=input_shape)
                
        if text_col == 'Title':
            # CNN for title text - simpler architecture
            x = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.5)(x)
            
            # Global pooling
            x = GlobalMaxPooling1D()(x)
            
            # Dense layers
            x = BatchNormalization()(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.5)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            # Create and compile model
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            # CNN for full text - more complex architecture
            # convolutional block
            x = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            
            # Global pooling
            x = GlobalMaxPooling1D()(x)
            
            # Dense layers
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            # Create and compile model
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.00005),  # Lower learning rate for more complex model
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        print("CNN model created on CPU")
        return model
    
    def create_lstm_model(self, text_col, input_shape):
        """
        Create an LSTM model for text classification with FinBERT embeddings.
        Different architectures for title vs full text.
        
        Args:
            text_col: The text column ('Title' or 'Full text')
            input_shape: Shape of the input data
            
        Returns:
            Compiled LSTM model
        """
        """Create a CPU-only LSTM model."""
        print("Creating LSTM model on CPU")
        inputs = Input(shape=input_shape)
                
        if text_col == 'Title':
            # LSTM for title text - simpler architecture
            x = LSTM(32, dropout=0.3, recurrent_dropout=0.3, return_sequences=False)(inputs)
            
            # Dense layers
            x = BatchNormalization()(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            # Create and compile model
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            # Regular LSTM for full text (not bidirectional)
            # LSTM layer
            x = LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=False)(inputs)
            x = BatchNormalization()(x)
            
            # Dense layers
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            # Create and compile model
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.00005),  # Lower learning rate for more complex model
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        print("LSTM model created on CPU")
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
            
            # Split data and get FinBERT embeddings
            (X_train_embeddings, X_train_mask, y_train), (X_val_embeddings, X_val_mask, y_val), (X_test_embeddings, X_test_mask, y_test), train_data, val_data, test_data = self.split_data(
                layer, text_col, label_col, model_type)
            
            # Check if there are enough samples and classes
            if len(X_train_embeddings) < 10 or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
                print(f"Skipping layer {i+1} due to insufficient data or class imbalance")
                continue
            
            # Get input shape - the FinBERT embeddings shape is (sequence_length, hidden_size)
            input_shape = (X_train_embeddings.shape[1], X_train_embeddings.shape[2])
            
            # Create model
            if model_type == 'CNN':
                model = self.create_cnn_model(text_col, input_shape)
            else:  # 'LSTM'
                model = self.create_lstm_model(text_col, input_shape)
                
            print(f"Created {model_type} model for {text_col} ({label_col})")
            model.summary()
            
            # Setup callbacks
            plot_callback = PlotLearningCallback(model_type, text_col, label_col, i+1, visualization_dir)
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
            model_checkpoint = ModelCheckpoint(
                filepath=f"{visualization_dir}/best_{model_type.lower()}_{text_col.lower().replace(' ', '_')}_{label_col}_layer_{i+1}.weights.h5",
                monitor='val_loss',
                save_weights_only=True,
                save_best_only=True,
                verbose=1
            )
            
            # Train model
            print(f"Training {model_type} model...")
            # Adjust batch size based on text length - smaller batch for full text to conserve memory
            batch_size = 32 if text_col == 'Title' else 16  
            history = model.fit(
                X_train_embeddings, y_train,
                validation_data=(X_val_embeddings, y_val),
                epochs=50,  
                batch_size=batch_size,
                callbacks=[early_stopping, model_checkpoint, plot_callback],
                verbose=1
            )
            
            # Evaluate on test set
            y_pred_proba = model.predict(X_test_embeddings)
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
    
    def clean_gpu_memory(self):
        """Clean GPU memory cache thoroughly."""
        # Clear all unreferenced cached memory
        torch.cuda.empty_cache()
    
        # Force garbage collection to clean up any remaining objects
        gc.collect()
    
        print("GPU memory cache cleaned")
    
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
        
        # clean GPU catch
        self.clean_gpu_memory()
        
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
        plt.title('Performance Comparison of CNN Models with FinBERT-ESG')
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
        save_path = os.path.join('Climate_news_second_database/FinBERT_Y_CNN_LSTM/visualizations_summary/COP', "cnn_performance_comparison.png")
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
        plt.title('Performance Comparison of LSTM Models with FinBERT-ESG')
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
        save_path = os.path.join('Climate_news_second_database/FinBERT_Y_CNN_LSTM/visualizations_summary/COP', "lstm_performance_comparison.png")
        plt.savefig(save_path)
        print(f"LSTM summary comparison visualization saved as: {save_path}")
        plt.close()
        
        # Layer-specific performance visualizations
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
        plt.title(f'{model_type} Accuracy by Model Combination and Layer with FinBERT-ESG')
        plt.ylim(0, max(0.8, df['Accuracy'].max() + 0.05))
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        # 2. AUC by combination and layer
        plt.subplot(2, 1, 2)
        sns.barplot(x='Combination', y='AUC', hue='Layer', data=df)
        plt.title(f'{model_type} AUC by Model Combination and Layer with FinBERT-ESG')
        plt.ylim(0, max(0.8, df['AUC'].max() + 0.05))
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        
        # Save the visualization
        save_path = os.path.join('Climate_news_second_database/FinBERT_Y_CNN_LSTM/visualizations_summary/COP', f"{model_type.lower()}_layer_performance.png")
        plt.savefig(save_path)
        print(f"{model_type} layer performance visualization saved as: {save_path}")
        plt.close()


# Main execution
if __name__ == "__main__":
    # Ensure visualization directories exist
    for directory in ['Climate_news_second_database/FinBERT_Y_CNN_LSTM/visualizations_cnn/COP', 'Climate_news_second_database/FinBERT_Y_CNN_LSTM/visualizations_lstm/COP', 
                      'Climate_news_second_database/FinBERT_Y_CNN_LSTM/visualizations_summary/COP']:
        os.makedirs(directory, exist_ok=True)
    
    csv_path = '/home/c.c24004260/Climate_news_second_database/us_news_semantics_COP_completed.csv'
    
    # Hugging Face token (optional)
    hf_token = None #"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"  # Could be set to None if not needed
    
    # Initialize the predictor with FinBERT model and token
    predictor = ClimateNewsFinBERTPredictor(csv_path, finbert_model="yiyanghkust/finbert-esg-9-categories", hf_token=hf_token)
    
    # Run the complete analysis
    predictor.load_data().define_time_windows().run_all_combinations()
    
    print("\nAnalysis complete! Results saved to 'FinBERT_Y_CNN_LSTM' directory.")
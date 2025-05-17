import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras import backend as K
import os
from datetime import datetime
import re
import ast
import warnings
warnings.filterwarnings('ignore')

# Define the focal loss function
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance in binary classification.
    
    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples.
            gamma=0 makes this equivalent to cross-entropy.
        alpha: Class weight parameter. For class imbalance. 
            0.5 means equal weighting for both classes.
    
    References:
        - "Focal Loss for Dense Object Detection" by Lin et al.
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to avoid numerical issues
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate pt (probability of true class)
        # For y_true=1 samples, pt = y_pred
        # For y_true=0 samples, pt = 1-y_pred
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        # Apply the focal loss formula:
        # -alpha * (1-pt)^gamma * log(pt)
        # Split into two terms for clarity
        loss_1 = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))
        loss_0 = -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        
        # Total loss is sum of both terms
        return loss_1 + loss_0
    
    return focal_loss_fixed

class Fino1SemanticVectorPredictor:
    def __init__(self, csv_path):
        """
        Initialize the stock predictor for Fino1 semantic vector analysis.
        
        Args:
            csv_path: Path to the CSV file containing Fino1 semantic vectors and stock labels
        """
        self.csv_path = csv_path
        self.data = None
        self.layers = []
        self.results = {}
        self.grid_search_results = {}  # Store grid search models and results
        
        # Define semantic vector columns - these are the exact column names in the CSV
        self.vector_columns = {
            'title': 'Fino1_title',
            'full_text': 'Fino1_full_text'
        }
        
        self.vector_dim = 19  # Dimension of semantic vectors
        self.parsed_data = {}  # Store parsed vectors
        self.scalers = {}  # Store normalizers for each dimension
        
        # Store optimal alpha values for focal loss (will be calculated from data)
        self.optimal_alpha = {
            'S_label': 0.5,  # Default, will be updated based on data
            'L_label': 0.5   # Default, will be updated based on data
        }
        
        # Create directories for visualizations
        os.makedirs('Climate_news_second_database/Fino1_semantic_LSTM_Plots/visualizations/COP/learning_curves', exist_ok=True)
        os.makedirs('Climate_news_second_database/Fino1_semantic_LSTM_Plots/visualizations/COP/performance_comparisons', exist_ok=True)
        os.makedirs('Climate_news_second_database/Fino1_semantic_LSTM_Plots/visualizations/COP/layer_comparisons', exist_ok=True)
        os.makedirs('Climate_news_second_database/Fino1_semantic_LSTM_Plots/visualizations/COP/confusion_matrices', exist_ok=True)
        
        # Create directory for Fino1 learning curves
        os.makedirs('Climate_news_second_database/Fino1_semantic_LSTM_Plots/visualizations/COP/learning_curves/Fino1', exist_ok=True)
        
        # Create directory for model checkpoints
        os.makedirs('Climate_news_second_database/Fino1_semantic_LSTM_Plots/Best_model/COP', exist_ok=True)
        
    
    def parse_vector(self, vector_str):
        """
        Parse a string representation of a vector into a numpy array.
        
        Args:
            vector_str: String representation of a vector
            
        Returns:
            Numpy array with the parsed vector values
        """
        try:
            # Handle different potential formats of vector strings
            if isinstance(vector_str, str):
                # Try using ast.literal_eval first for safer parsing
                try:
                    vector_data = ast.literal_eval(vector_str)
                    # Check if it's a nested list like [[1, 2, 3, ...]]
                    if isinstance(vector_data, list) and len(vector_data) == 1 and isinstance(vector_data[0], list):
                        return np.array(vector_data[0])
                    # Check if it's a single list like [1, 2, 3, ...]
                    elif isinstance(vector_data, list):
                        return np.array(vector_data)
                except (ValueError, SyntaxError):
                    # Fallback to regex-based parsing
                    # Remove all brackets and split by commas
                    clean_str = re.sub(r'[\[\]]', '', vector_str)
                    values = [float(x.strip()) for x in clean_str.split(',') if x.strip()]
                    return np.array(values)
            else:
                # If it's already a list or array, convert to numpy array
                return np.array(vector_str)
        except Exception as e:
            print(f"Error parsing vector: {e}")
            print(f"Problematic vector string: {vector_str}")
            # Return zero vector as fallback
            return np.zeros(self.vector_dim)
    
    def load_data(self):
        """Load and preprocess the CSV data."""
        print("Loading data...")
        self.data = pd.read_csv(self.csv_path)
        
        # Convert date strings to datetime objects
        try:
            # Try pandas' automatic date parsing first with dayfirst=True
            self.data['Publication date'] = pd.to_datetime(self.data['Publication date'], dayfirst=True)
            self.data['Predicting date Short'] = pd.to_datetime(self.data['Predicting date Short'], dayfirst=True)
            self.data['Predicting date Long'] = pd.to_datetime(self.data['Predicting date Long'], dayfirst=True)
        except (ValueError, TypeError):
            print("Automatic date parsing failed. Trying manual conversion...")
            
            # Helper function to handle different date formats
            def parse_date_column(column):
                result = []
                for date_str in column:
                    try:
                        # Try to parse as DD/MM/YYYY
                        date = pd.to_datetime(date_str, format='%d/%m/%Y')
                    except ValueError:
                        try:
                            # Try to parse as YYYY-MM-DD
                            date = pd.to_datetime(date_str, format='%Y-%m-%d')
                        except ValueError:
                            # As a last resort, let pandas guess with dayfirst=True
                            date = pd.to_datetime(date_str, dayfirst=True)
                    result.append(date)
                return pd.Series(result)
            
            self.data['Publication date'] = parse_date_column(self.data['Publication date'])
            self.data['Predicting date Short'] = parse_date_column(self.data['Predicting date Short'])
            self.data['Predicting date Long'] = parse_date_column(self.data['Predicting date Long'])
        
        # Sort by publication date
        self.data = self.data.sort_values('Publication date')
        
        # Display initial data info
        print(f"Loaded {len(self.data)} financial news articles spanning from "
              f"{self.data['Publication date'].min().strftime('%d/%m/%Y')} "
              f"to {self.data['Publication date'].max().strftime('%d/%m/%Y')}")
        
        # Analyze class distributions and calculate optimal alpha values for focal loss
        s_counts = self.data['S_label'].value_counts()
        l_counts = self.data['L_label'].value_counts()
        
        print(f"Class distribution for short-term prediction: {s_counts.to_dict()}")
        print(f"Class distribution for long-term prediction: {l_counts.to_dict()}")
        
        # Calculate class imbalance ratios and suggested alpha values
        if 1 in s_counts and 0 in s_counts:
            s_ratio = s_counts[1] / len(self.data)
            self.optimal_alpha['S_label'] = 1 - s_ratio  # Set alpha inversely proportional to class frequency
            print(f"\nShort-term class 1 ratio: {s_ratio:.4f} (suggested alpha: {self.optimal_alpha['S_label']:.4f})")
        
        if 1 in l_counts and 0 in l_counts:
            l_ratio = l_counts[1] / len(self.data)
            self.optimal_alpha['L_label'] = 1 - l_ratio
            print(f"Long-term class 1 ratio: {l_ratio:.4f} (suggested alpha: {self.optimal_alpha['L_label']:.4f})")
        
        # Check if the required columns exist
        for col_name in self.vector_columns.values():
            if col_name not in self.data.columns:
                raise ValueError(f"Required column '{col_name}' not found in the CSV file.")
        
        # Parse vector columns
        self._parse_vector_columns()
        
        # Normalize semantic vectors
        self._normalize_semantic_vectors()
        
        return self
    
    def _parse_vector_columns(self):
        """Parse all semantic vector columns and store them."""
        print("\nParsing semantic vector columns...")
        
        for text_type, col in self.vector_columns.items():
            print(f"Parsing {col}...")
            
            # Create a new column in parsed_data to store the parsed vectors
            self.parsed_data[col] = []
            
            # Parse each vector string
            for i, vec_str in enumerate(self.data[col]):
                try:
                    parsed_vec = self.parse_vector(vec_str)
                    
                    # Ensure the vector has the expected dimension
                    if len(parsed_vec) != self.vector_dim:
                        print(f"Warning: Vector at index {i} has {len(parsed_vec)} dimensions instead of {self.vector_dim}.")
                        # Pad or truncate to ensure consistent dimensions
                        if len(parsed_vec) < self.vector_dim:
                            # Pad with zeros
                            parsed_vec = np.pad(parsed_vec, (0, self.vector_dim - len(parsed_vec)))
                        else:
                            # Truncate
                            parsed_vec = parsed_vec[:self.vector_dim]
                    
                    self.parsed_data[col].append(parsed_vec)
                except Exception as e:
                    print(f"Error parsing vector at index {i}: {e}")
                    # Use zero vector as fallback
                    self.parsed_data[col].append(np.zeros(self.vector_dim))
            
            # Convert list of vectors to a 2D numpy array
            self.parsed_data[col] = np.array(self.parsed_data[col])
            
            # Basic statistics of parsed vectors
            print(f"  Shape: {self.parsed_data[col].shape}")
            print(f"  Mean: {np.mean(self.parsed_data[col]):.4f}")
            print(f"  Min: {np.min(self.parsed_data[col]):.4f}")
            print(f"  Max: {np.max(self.parsed_data[col]):.4f}")
    
    def _normalize_semantic_vectors(self):
        """Normalize semantic vectors using dimension-wise Min-Max scaling."""
        print("\nNormalizing semantic vectors...")
        
        for text_type, col in self.vector_columns.items():
            print(f"Normalizing {col}...")
            
            if col not in self.parsed_data or len(self.parsed_data[col]) == 0:
                print(f"  Warning: No parsed data found for {col}. Skipping normalization.")
                continue
            
            vectors = self.parsed_data[col]
            normalized_vectors = np.zeros_like(vectors, dtype=np.float32)
            dim_scalers = []
            
            # Apply Min-Max scaling to each dimension independently
            for dim in range(self.vector_dim):
                scaler = MinMaxScaler()
                normalized_vectors[:, dim] = scaler.fit_transform(vectors[:, dim].reshape(-1, 1)).flatten()
                dim_scalers.append(scaler)
            
            # Store normalized vectors and scalers
            self.parsed_data[f"{col}_norm"] = normalized_vectors
            self.scalers[col] = dim_scalers
            
            # Print statistics of normalized vectors
            print(f"  Normalized shape: {normalized_vectors.shape}")
            print(f"  Normalized mean: {np.mean(normalized_vectors):.4f}")
            print(f"  Normalized min: {np.min(normalized_vectors):.4f}")
            print(f"  Normalized max: {np.max(normalized_vectors):.4f}")
        
        return self
    
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
    
    def split_data(self, layer, vector_column, label_column):
        """
        Split the data into training, validation, and test sets based on the defined time windows.
        
        Args:
            layer: The time window layer
            vector_column: The column containing the semantic vectors
            label_column: The column containing the labels ('S_label' or 'L_label')
            
        Returns:
            Split datasets and corresponding indices
        """
        # Create masks for each time period
        train_mask = (self.data['Publication date'] >= layer['train_start']) & (self.data['Publication date'] <= layer['train_end'])
        val_mask = (self.data['Publication date'] > layer['val_start']) & (self.data['Publication date'] <= layer['val_end'])
        test_mask = (self.data['Publication date'] > layer['test_start']) & (self.data['Publication date'] <= layer['test_end'])
        
        # Get indices for each period
        train_indices = self.data[train_mask].index
        val_indices = self.data[val_mask].index
        test_indices = self.data[test_mask].index
        
        print(f"Training data: {len(train_indices)} samples")
        print(f"Validation data: {len(val_indices)} samples")
        print(f"Test data: {len(test_indices)} samples")
        
        # Get normalized vector data using the indices
        norm_col = f"{vector_column}_norm"
        X_train = self.parsed_data[norm_col][np.where(np.isin(self.data.index, train_indices))[0]]
        X_val = self.parsed_data[norm_col][np.where(np.isin(self.data.index, val_indices))[0]]
        X_test = self.parsed_data[norm_col][np.where(np.isin(self.data.index, test_indices))[0]]
        
        # Get labels
        y_train = self.data.loc[train_indices, label_column].values
        y_val = self.data.loc[val_indices, label_column].values
        y_test = self.data.loc[test_indices, label_column].values
        
        # Analyze class distribution in each split for focal loss parameter tuning
        train_class_dist = np.bincount(y_train.astype(int)) if len(y_train) > 0 else [0, 0]
        val_class_dist = np.bincount(y_val.astype(int)) if len(y_val) > 0 else [0, 0]
        test_class_dist = np.bincount(y_test.astype(int)) if len(y_test) > 0 else [0, 0]
        
        # Ensure distributions have two elements (for binary classification)
        if len(train_class_dist) < 2:
            train_class_dist = np.pad(train_class_dist, (0, 2 - len(train_class_dist)))
        if len(val_class_dist) < 2:
            val_class_dist = np.pad(val_class_dist, (0, 2 - len(val_class_dist)))
        if len(test_class_dist) < 2:
            test_class_dist = np.pad(test_class_dist, (0, 2 - len(test_class_dist)))
            
        print(f"Train class distribution: [Class 0: {train_class_dist[0]}, Class 1: {train_class_dist[1]}]")
        print(f"Val class distribution: [Class 0: {val_class_dist[0]}, Class 1: {val_class_dist[1]}]")
        print(f"Test class distribution: [Class 0: {test_class_dist[0]}, Class 1: {test_class_dist[1]}]")
        
        # Calculate optimal alpha for focal loss based on training set
        train_class_ratio = train_class_dist[1] / (train_class_dist[0] + train_class_dist[1]) if (train_class_dist[0] + train_class_dist[1]) > 0 else 0.5
        optimal_alpha = 1 - train_class_ratio  # Inverse of class 1 ratio
        
        print(f"Train set class 1 ratio: {train_class_ratio:.4f}")
        print(f"Calculated optimal alpha for focal loss: {optimal_alpha:.4f}")
        
        return (X_train, y_train, train_indices), (X_val, y_val, val_indices), (X_test, y_test, test_indices), optimal_alpha
    
    def create_sequence_data(self, X, y, indices, window_size=10):
        """
        Create sequences for LSTM by sliding a window over the time series data.
        Each sequence will preserve the vector dimensions.
        
        Args:
            X: Feature array (semantic vectors) with shape (n_samples, vector_dim)
            y: Target array (stock trend labels)
            indices: Indices of the data points in the original dataframe
            window_size: The number of previous time steps to include
            
        Returns:
            X_seq, y_seq: Sequences for LSTM input and corresponding targets
        """
        X_seq, y_seq = [], []
        
        # Get publication dates for the indices
        pub_dates = self.data.loc[indices, 'Publication date'].reset_index(drop=True)
        
        # Calculate the time difference between consecutive records
        time_diffs = pub_dates.diff().dt.days
        
        # Replace NaN with 0 for the first element
        time_diffs.iloc[0] = 0
        
        # Create a cumulative sequence counter
        sequence_id = 0
        seq_counters = [0]
        
        for i in range(1, len(time_diffs)):
            # If the time difference is too large (e.g., more than 30 days), 
            # it indicates a gap in the time series, so start a new sequence
            if time_diffs.iloc[i] > 30:
                sequence_id += 1
            seq_counters.append(sequence_id)
        
        # Process each sequence separately
        unique_sequences = np.unique(seq_counters)
        
        for seq_id in unique_sequences:
            seq_mask = [i for i, x in enumerate(seq_counters) if x == seq_id]
            X_sub = X[seq_mask]  # Shape: (n_samples, vector_dim)
            y_sub = y[seq_mask]
            
            # Create sequences only if we have enough data points
            if len(X_sub) >= window_size:
                for i in range(len(X_sub) - window_size):
                    # Each sequence will have shape (window_size, vector_dim)
                    X_seq.append(X_sub[i:i+window_size])
                    y_seq.append(y_sub[i+window_size])
        
        # Convert to numpy arrays
        X_seq_array = np.array(X_seq)
        y_seq_array = np.array(y_seq)
        
        # Analyze class distribution in sequences for focal loss parameter tuning
        if len(y_seq_array) > 0:
            seq_class_dist = np.bincount(y_seq_array.astype(int))
            
            # Ensure distribution has two elements
            if len(seq_class_dist) < 2:
                seq_class_dist = np.pad(seq_class_dist, (0, 2 - len(seq_class_dist)))
                
            print(f"Sequence class distribution: [Class 0: {seq_class_dist[0]}, Class 1: {seq_class_dist[1]}]")
            
            # Calculate optimal alpha based on sequence class distribution
            seq_class_ratio = seq_class_dist[1] / len(y_seq_array) if len(y_seq_array) > 0 else 0.5
            seq_optimal_alpha = 1 - seq_class_ratio
            
            print(f"Sequence class 1 ratio: {seq_class_ratio:.4f}")
            print(f"Sequence optimal focal loss alpha: {seq_optimal_alpha:.4f}")
            
            return X_seq_array, y_seq_array, seq_optimal_alpha
        
        # Default return if no sequences could be created
        return X_seq_array, y_seq_array, 0.5
    
    def create_lstm_model(self, window_size, lstm_units=32, dropout_rate=0.5, 
                          learning_rate=0.1, lstm_layers=1, focal_alpha=0.5, focal_gamma=2.0):
        """
        Create an LSTM model for sequence prediction with semantic vectors and focal loss.
        
        Args:
            window_size: Number of time steps in each sequence
            lstm_units: Number of units in LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
            lstm_layers: Number of LSTM layers (1 or 2)
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        if lstm_layers == 1:
            model.add(LSTM(lstm_units, input_shape=(window_size, self.vector_dim), 
                          dropout=dropout_rate, recurrent_dropout=dropout_rate/2))
        else:
            model.add(LSTM(lstm_units, input_shape=(window_size, self.vector_dim), 
                          dropout=dropout_rate, recurrent_dropout=dropout_rate/2,
                          return_sequences=True))
            model.add(LSTM(lstm_units // 2, dropout=dropout_rate, recurrent_dropout=dropout_rate/2))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Create focal loss function
        fl = focal_loss(gamma=focal_gamma, alpha=focal_alpha)
        
        # Compile model with focal loss
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=fl,
            metrics=['accuracy', 
                     tf.keras.metrics.AUC(name='auc'),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def train_and_evaluate(self, vector_column, label_column, window_size=10,
                          lstm_units=32, dropout_rate=0.5, learning_rate=0.1,
                          batch_size=32, epochs=100, lstm_layers=1,
                          focal_alpha=None, focal_gamma=2.0,
                          store_for_reuse=False):
        """
        Train and evaluate LSTM model for a specific semantic vector column and label column
        using focal loss to address class imbalance.
        
        Args:
            vector_column: The semantic vector column to use
            label_column: The label column to use ('S_label' or 'L_label')
            window_size: Number of previous time steps to include in each sequence
            lstm_units: Number of units in LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            lstm_layers: Number of LSTM layers (1 or 2)
            focal_alpha: Alpha parameter for focal loss (if None, calculated from data)
            focal_gamma: Gamma parameter for focal loss
            store_for_reuse: Whether to store model results for later reuse
            
        Returns:
            Dictionary with layer results if store_for_reuse is True
        """
        # Determine text type (title or full_text) from the vector column name
        text_type = 'Title' if 'title' in vector_column.lower() else 'Full_text'
        
        # Determine label type
        if label_column == 'S_label':
            label_type = 'Short-term'
        else:
            label_type = 'Long-term'
        
        # If focal_alpha is not provided, use the optimal alpha for this label type
        if focal_alpha is None:
            focal_alpha = self.optimal_alpha.get(label_column, 0.5)
            print(f"Using calculated optimal focal loss alpha: {focal_alpha:.4f}")
        
        # Store results
        combination_key = f"Fino1_{text_type}_{label_type}"
        self.results[combination_key] = {
            'accuracy': [],
            'auc': [],
            'f1_score': [],  # Added F1 score tracking
            'precision': [],  # Added precision tracking
            'recall': [],     # Added recall tracking
            'layer_results': [],
            'window_size': window_size,  # Store window size with results
            'focal_alpha': focal_alpha,  # Store focal loss parameters
            'focal_gamma': focal_gamma
        }
        
        # Used to store complete results for all layers if we need to reuse them
        all_layer_results = []
        
        print(f"\n{'='*80}")
        print(f"Training model for Fino1 approach, {text_type}, {label_type}")
        print(f"Using focal loss with alpha={focal_alpha:.4f}, gamma={focal_gamma:.2f}")
        print(f"{'='*80}")
        
        for i, layer in enumerate(self.layers):
            print(f"\nLayer {i+1}:")
            print(f"Training period: {layer['train_start'].strftime('%d/%m/%Y')} - {layer['train_end'].strftime('%d/%m/%Y')}")
            print(f"Validation period: {layer['val_start'].strftime('%d/%m/%Y')} - {layer['val_end'].strftime('%d/%m/%Y')}")
            print(f"Testing period: {layer['test_start'].strftime('%d/%m/%Y')} - {layer['test_end'].strftime('%d/%m/%Y')}")
            
            # Split data for this layer
            (X_train, y_train, train_indices), \
            (X_val, y_val, val_indices), \
            (X_test, y_test, test_indices), \
            split_alpha = self.split_data(layer, vector_column, label_column)
            
            # Check if there are enough samples and classes
            if len(X_train) < window_size + 1 or len(np.unique(y_train)) < 2 or \
               len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
                print(f"Skipping layer {i+1} due to insufficient data or class imbalance")
                continue
            
            # Create sequence data and get sequence-specific alpha
            X_train_seq, y_train_seq, seq_alpha = self.create_sequence_data(X_train, y_train, train_indices, window_size)
            X_val_seq, y_val_seq, _ = self.create_sequence_data(X_val, y_val, val_indices, window_size)
            X_test_seq, y_test_seq, _ = self.create_sequence_data(X_test, y_test, test_indices, window_size)
            
            # Use the sequence-specific alpha if it's more balanced than the provided one
            # This helps with cases where sequence creation changes class balance
            current_alpha = seq_alpha if abs(seq_alpha - 0.5) < abs(focal_alpha - 0.5) else focal_alpha
            print(f"Using focal loss alpha={current_alpha:.4f} for this layer")
            
            # Check if we have enough sequence data
            if len(X_train_seq) < 10 or len(X_val_seq) < 5 or len(X_test_seq) < 5:
                print(f"Skipping layer {i+1} due to insufficient sequence data")
                continue
            
            print(f"Training sequences: {len(X_train_seq)}")
            print(f"Validation sequences: {len(X_val_seq)}")
            print(f"Testing sequences: {len(X_test_seq)}")
            
            # Create and train model with focal loss
            model = self.create_lstm_model(
                window_size=window_size,
                lstm_units=lstm_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                lstm_layers=lstm_layers,
                focal_alpha=current_alpha,
                focal_gamma=focal_gamma
            )
            
            # Define callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ModelCheckpoint(
                    f'Climate_news_second_database/Fino1_semantic_LSTM_Plots/Best_model/COP/model_Fino1_SemanticVectors_{text_type}_{label_type}_focal_layer{i+1}.h5',
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            # Train model
            history = model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate on test set
            y_pred_prob = model.predict(X_test_seq)
            
            # Find optimal threshold based on F1 score on validation set
            if len(y_val_seq) > 0 and len(np.unique(y_val_seq)) > 1:
                val_pred_prob = model.predict(X_val_seq)
                precisions, recalls, thresholds = precision_recall_curve(y_val_seq, val_pred_prob)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                print(f"Optimal threshold based on F1 score: {optimal_threshold:.4f}")
            else:
                optimal_threshold = 0.5
                print("Using default threshold of 0.5 (couldn't calculate optimal threshold)")
            
            # Apply optimal threshold to test predictions
            y_pred = (y_pred_prob > optimal_threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_seq, y_pred)
            
            # Only calculate other metrics if we have both classes in the test set
            if len(np.unique(y_test_seq)) > 1:
                auc = roc_auc_score(y_test_seq, y_pred_prob)
                f1 = f1_score(y_test_seq, y_pred)
                precision = precision_score(y_test_seq, y_pred)
                recall = recall_score(y_test_seq, y_pred)
            else:
                print("Warning: Test set has only one class, some metrics cannot be calculated")
                auc = 0.5  # Default for random performance
                f1 = 0.0 if np.all(y_test_seq == 0) else 1.0  # All negatives or all positives
                precision = 0.0 if np.all(y_test_seq == 0) else 1.0
                recall = 0.0 if np.all(y_test_seq == 0) else 1.0
            
            # Store metrics
            self.results[combination_key]['accuracy'].append(accuracy)
            self.results[combination_key]['auc'].append(auc)
            self.results[combination_key]['f1_score'].append(f1)
            self.results[combination_key]['precision'].append(precision)
            self.results[combination_key]['recall'].append(recall)
            
            print(f"Test Accuracy for Layer {i+1}: {accuracy:.4f}")
            print(f"Test AUC for Layer {i+1}: {auc:.4f}")
            print(f"Test F1 Score for Layer {i+1}: {f1:.4f}")
            print(f"Test Precision for Layer {i+1}: {precision:.4f}")
            print(f"Test Recall for Layer {i+1}: {recall:.4f}")
            
            # Create confusion matrix
            cm = confusion_matrix(y_test_seq, y_pred)
            print("Confusion Matrix:")
            print(cm)
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(y_test_seq, y_pred))
            
            # Visualize confusion matrix
            self.visualize_confusion_matrix(
                cm,
                'Fino1',
                text_type,
                label_type,
                i+1,
                focal_alpha=current_alpha,
                focal_gamma=focal_gamma
            )
            
            # Store layer results for visualization
            layer_result = {
                'layer': i+1,
                'history': history.history,
                'model': model if store_for_reuse else None,  # Only store model if needed for reuse
                'X_train_seq': X_train_seq if store_for_reuse else None,
                'y_train_seq': y_train_seq if store_for_reuse else None,
                'X_val_seq': X_val_seq if store_for_reuse else None,
                'y_val_seq': y_val_seq if store_for_reuse else None,
                'X_test_seq': X_test_seq if store_for_reuse else None,
                'y_test_seq': y_test_seq if store_for_reuse else None,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'confusion_matrix': cm,
                'accuracy': accuracy,
                'auc': auc,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'window_size': window_size,
                'focal_alpha': current_alpha,
                'focal_gamma': focal_gamma,
                'optimal_threshold': optimal_threshold
            }
            
            self.results[combination_key]['layer_results'].append(layer_result)
            
            if store_for_reuse:
                all_layer_results.append(layer_result)
            
            # Visualize learning curves for this layer
            self.visualize_learning_curves(
                history.history,
                'Fino1',
                text_type,
                label_type,
                i+1,
                focal_alpha=current_alpha,
                focal_gamma=focal_gamma
            )
        
        # Calculate average metrics
        if self.results[combination_key]['accuracy']:
            avg_accuracy = np.mean(self.results[combination_key]['accuracy'])
            avg_auc = np.mean(self.results[combination_key]['auc'])
            avg_f1 = np.mean(self.results[combination_key]['f1_score'])
            avg_precision = np.mean(self.results[combination_key]['precision'])
            avg_recall = np.mean(self.results[combination_key]['recall'])
            
            print(f"\nAverage Test Accuracy across all layers: {avg_accuracy:.4f}")
            print(f"Average Test AUC across all layers: {avg_auc:.4f}")
            print(f"Average Test F1 Score across all layers: {avg_f1:.4f}")
            print(f"Average Test Precision across all layers: {avg_precision:.4f}")
            print(f"Average Test Recall across all layers: {avg_recall:.4f}")
            
            self.results[combination_key]['avg_accuracy'] = avg_accuracy
            self.results[combination_key]['avg_auc'] = avg_auc
            self.results[combination_key]['avg_f1_score'] = avg_f1
            self.results[combination_key]['avg_precision'] = avg_precision
            self.results[combination_key]['avg_recall'] = avg_recall
        else:
            print("\nNo valid layers to calculate average metrics")
            self.results[combination_key]['avg_accuracy'] = np.nan
            self.results[combination_key]['avg_auc'] = np.nan
            self.results[combination_key]['avg_f1_score'] = np.nan
            self.results[combination_key]['avg_precision'] = np.nan
            self.results[combination_key]['avg_recall'] = np.nan
        
        if store_for_reuse:
            return {
                'combination_key': combination_key,
                'layer_results': all_layer_results,
                'avg_accuracy': self.results[combination_key].get('avg_accuracy', np.nan),
                'avg_auc': self.results[combination_key].get('avg_auc', np.nan),
                'avg_f1_score': self.results[combination_key].get('avg_f1_score', np.nan),
                'avg_precision': self.results[combination_key].get('avg_precision', np.nan),
                'avg_recall': self.results[combination_key].get('avg_recall', np.nan),
                'window_size': window_size,
                'focal_alpha': focal_alpha,
                'focal_gamma': focal_gamma
            }
        else:
            return None
    
    def visualize_confusion_matrix(self, cm, approach, text_type, label_type, layer_num, 
                                  focal_alpha=None, focal_gamma=None):
        """
        Visualize confusion matrix for a specific model and layer.
        
        Args:
            cm: Confusion matrix
            approach: Sentiment analysis approach (always 'Fino1')
            text_type: Text type (Title or Full_text)
            label_type: Label type (Short-term or Long-term)
            layer_num: Layer number
            focal_alpha: Alpha parameter used for focal loss
            focal_gamma: Gamma parameter used for focal loss
        """
        plt.figure(figsize=(8, 6))
        
        # Create a normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] if cm.sum(axis=1).any() else cm
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaNs with zeros
        
        # Create heatmap with both raw counts and percentages
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        
        # Add focal loss parameters to title if provided
        focal_params = ''
        if focal_alpha is not None and focal_gamma is not None:
            focal_params = f" (Focal Loss α={focal_alpha:.2f}, γ={focal_gamma:.1f})"
        
        plt.title(f"Confusion Matrix: Fino1 Semantic Vectors - {text_type} - {label_type}\nLayer {layer_num}{focal_params}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save figure
        save_path = f'Climate_news_second_database/Fino1_semantic_LSTM_Plots/visualizations/COP/confusion_matrices/Fino1_{text_type}_{label_type}_layer{layer_num}_focal.png'
        plt.savefig(save_path)
        plt.close()
    
    def visualize_learning_curves(self, history, approach, text_type, label_type, layer_num,
                                 focal_alpha=None, focal_gamma=None):
        """
        Visualize learning curves for a specific model and layer.
        
        Args:
            history: Training history dictionary
            approach: Sentiment analysis approach (always 'Fino1')
            text_type: Text type (Title or Full_text)
            label_type: Label type (Short-term or Long-term)
            layer_num: Layer number
            focal_alpha: Alpha parameter used for focal loss
            focal_gamma: Gamma parameter used for focal loss
        """
        # Create a figure with multiple subplots for different metrics
        plt.figure(figsize=(15, 10))
        
        # Add focal loss parameters to title if provided
        focal_params = ''
        if focal_alpha is not None and focal_gamma is not None:
            focal_params = f" (Focal Loss α={focal_alpha:.2f}, γ={focal_gamma:.1f})"
        
        plt.suptitle(f"Learning Curves: {approach} Semantic Vectors - {text_type} - {label_type}\nLayer {layer_num}{focal_params}", 
                   fontsize=16)
        
        # Determine which metrics are available
        available_metrics = [metric for metric in history.keys() if not metric.startswith('val_')]
        
        # Create a subplot for each metric
        for i, metric in enumerate(available_metrics):
            plt.subplot(2, (len(available_metrics) + 1) // 2, i + 1)
            plt.plot(history[metric], label=f'Training {metric}')
            plt.plot(history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'{metric.capitalize()} Curves')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Add focal to filename if using focal loss
        focal_suffix = "_focal" if focal_alpha is not None else ""
        
        save_path = f'Climate_news_second_database/Fino1_semantic_LSTM_Plots/visualizations/COP/learning_curves/{approach}/{text_type}_{label_type}_SemanticVectors_layer{layer_num}{focal_suffix}.png'
        plt.savefig(save_path)
        plt.close()
    
    def run_grid_search(self, vector_column, label_column):
        """
        Run a grid search for best window size and focal loss parameters.
        
        Args:
            vector_column: The semantic vector column to use
            label_column: The label column to use
            
        Returns:
            Dictionary with best parameters and stored model results
        """
        # Window sizes to try
        window_sizes = [5, 10, 15, 20]
        
        # Get the default alpha for this label type
        default_alpha = self.optimal_alpha.get(label_column, 0.5)
        
        # Keep track of best parameters and results
        best_f1 = 0  # Using F1 score for selection (better for imbalanced data)
        best_params = {}
        best_model_results = None
        
        # Store all results by window size
        all_window_results = {}
        
        for window_size in window_sizes:
            print(f"\n{'='*80}")
            print(f"Testing window size: {window_size} with focal loss")
            print(f"{'='*80}")
            
            # Train and store full results for reuse
            model_results = self.train_and_evaluate(
                vector_column=vector_column,
                label_column=label_column,
                window_size=window_size,
                focal_alpha=default_alpha,  # Use calculated alpha for this label type
                focal_gamma=2.0,            # Standard gamma value
                store_for_reuse=True        # Store complete model results
            )
            
            if model_results:
                all_window_results[window_size] = model_results
                
                # Use F1 score for parameter selection (better for imbalanced data)
                current_f1 = model_results.get('avg_f1_score', 0)
                
                if not np.isnan(current_f1) and current_f1 > best_f1:
                    best_f1 = current_f1
                    best_params = {
                        'window_size': window_size,
                        'focal_alpha': default_alpha,
                        'focal_gamma': 2.0
                    }
                    best_model_results = model_results
                    print(f"New best F1 score: {best_f1:.4f} with window_size={window_size}, "
                         f"focal_alpha={default_alpha:.4f}")
        
        if best_params:
            best_window_size = best_params['window_size']
            print(f"\nBest window size: {best_window_size} with F1 score: {best_f1:.4f}")
            print(f"Best focal loss parameters: alpha={best_params.get('focal_alpha', 0.5):.4f}, "
                 f"gamma={best_params.get('focal_gamma', 2.0):.1f}")
        else:
            # If no valid results were found, default to window size 10
            best_window_size = 10
            print(f"\nNo valid results found. Defaulting to window size: {best_window_size}")
            best_params = {
                'window_size': best_window_size,
                'focal_alpha': default_alpha,
                'focal_gamma': 2.0
            }
        
        # Store in grid search results
        key = f"{vector_column}_{label_column}"
        self.grid_search_results[key] = {
            'best_params': best_params,
            'best_model_results': best_model_results,
            'all_window_results': all_window_results,
            'best_f1': best_f1
        }
        
        return best_params
    
    def run_all_combinations(self, perform_grid_search=True, focal_gamma=2.0):
        """
        Run the analysis for all combinations of Fino1 semantic vector columns and label columns.
        
        Args:
            perform_grid_search: Whether to perform grid search for window size first
            focal_gamma: Gamma parameter for focal loss
        """
        # Define all combinations for Fino1 semantic vector columns
        combinations = []
        
        for text_type, col in self.vector_columns.items():
            for label_col in ['S_label', 'L_label']:
                combinations.append((col, label_col))
        
        # Dictionary to store best window sizes and focal loss parameters
        best_params = {}
        
        # Run analysis for each combination
        for vector_col, label_col in combinations:
            # Get the default alpha for this label type
            default_alpha = self.optimal_alpha.get(label_col, 0.5)
            
            if perform_grid_search:
                print(f"\n{'='*80}")
                print(f"PERFORMING GRID SEARCH FOR {vector_col} - {label_col}")
                print(f"{'='*80}")
                
                # Run grid search to find the best window size
                grid_params = self.run_grid_search(vector_col, label_col)
                best_window_size = grid_params.get('window_size', 10)
                best_alpha = grid_params.get('focal_alpha', default_alpha)
                
                best_params[(vector_col, label_col)] = {
                    'window_size': best_window_size,
                    'focal_alpha': best_alpha,
                    'focal_gamma': focal_gamma
                }
                
                # No need to retrain - reuse the grid search results
                print(f"\n{'='*80}")
                print(f"USING BEST MODEL FOR {vector_col} - {label_col} WITH WINDOW SIZE {best_window_size}")
                print(f"FOCAL LOSS PARAMETERS: ALPHA={best_alpha:.4f}, GAMMA={focal_gamma:.1f}")
                print(f"{'='*80}")
                
                # Get the results structure
                key = f"{vector_col}_{label_col}"
                grid_result = self.grid_search_results.get(key, {})
                
                # Update self.results with the best model results from grid search
                if grid_result and 'best_model_results' in grid_result and grid_result['best_model_results']:
                    best_model_results = grid_result['best_model_results']
                    
                    # Determine text_type and label_type for combination_key
                    result_text_type = 'Title' if 'title' in vector_col.lower() else 'Full_text'
                    result_label_type = 'Short-term' if label_col == 'S_label' else 'Long-term'
                    combination_key = f"Fino1_{result_text_type}_{result_label_type}"
                    
                    # Update self.results with the best model results
                    self.results[combination_key] = {
                        'accuracy': [],
                        'auc': [],
                        'f1_score': [],
                        'precision': [],
                        'recall': [],
                        'avg_accuracy': best_model_results.get('avg_accuracy', np.nan),
                        'avg_auc': best_model_results.get('avg_auc', np.nan),
                        'avg_f1_score': best_model_results.get('avg_f1_score', np.nan),
                        'avg_precision': best_model_results.get('avg_precision', np.nan),
                        'avg_recall': best_model_results.get('avg_recall', np.nan),
                        'layer_results': best_model_results.get('layer_results', []),
                        'window_size': best_window_size,
                        'focal_alpha': best_alpha,
                        'focal_gamma': focal_gamma
                    }
                    
                    # Update metric arrays
                    for layer_result in best_model_results.get('layer_results', []):
                        self.results[combination_key]['accuracy'].append(layer_result.get('accuracy', np.nan))
                        self.results[combination_key]['auc'].append(layer_result.get('auc', np.nan))
                        self.results[combination_key]['f1_score'].append(layer_result.get('f1_score', np.nan))
                        self.results[combination_key]['precision'].append(layer_result.get('precision', np.nan))
                        self.results[combination_key]['recall'].append(layer_result.get('recall', np.nan))
                    
                    print(f"Updated results with best model (window_size={best_window_size})")
                    print(f"  Accuracy: {best_model_results.get('avg_accuracy', np.nan):.4f}")
                    print(f"  AUC: {best_model_results.get('avg_auc', np.nan):.4f}")
                    print(f"  F1 Score: {best_model_results.get('avg_f1_score', np.nan):.4f}")
                    print(f"  Precision: {best_model_results.get('avg_precision', np.nan):.4f}")
                    print(f"  Recall: {best_model_results.get('avg_recall', np.nan):.4f}")
            else:
                # Just train with default window size and focal loss
                print(f"\n{'='*80}")
                print(f"TRAINING MODEL FOR {vector_col} - {label_col} WITH FOCAL LOSS")
                print(f"{'='*80}")
                
                self.train_and_evaluate(
                    vector_column=vector_col, 
                    label_column=label_col,
                    window_size=10,
                    focal_alpha=default_alpha,
                    focal_gamma=focal_gamma
                )
        
        # Create summary visualizations
        self.create_summary_visualizations()
        
        # Print final summary
        self.print_summary()
        
        # Print best parameters
        if perform_grid_search:
            print("\n" + "="*80)
            print("BEST PARAMETERS BY COMBINATION")
            print("="*80)
            
            for (vector_col, label_col), params in best_params.items():
                text_type = 'Title' if 'title' in vector_col.lower() else 'Full_text'
                label_type = 'Short-term' if label_col == 'S_label' else 'Long-term'
                print(f"{text_type} + {label_type}: Window Size={params['window_size']}, "
                     f"Focal Alpha={params['focal_alpha']:.4f}, Focal Gamma={params['focal_gamma']:.1f}")
        
        return self
    
    def create_summary_visualizations(self):
        """Create summary visualizations comparing model performances."""
        self._create_performance_by_approach_visualizations()
        self._create_performance_by_layer_visualizations()
    
    def _create_performance_by_approach_visualizations(self):
        """Create visualizations comparing performance by text type and prediction timeframe."""
        # Collect data for Fino1 semantic vectors
        combinations = []
        accuracies = []
        aucs = []
        f1_scores = []
        precisions = []
        recalls = []
        window_sizes = []
        focal_alphas = []
        
        for text_type in ['Title', 'Full_text']:
            for label_type in ['Short-term', 'Long-term']:
                combination_key = f"Fino1_{text_type}_{label_type}"
                
                if combination_key in self.results and 'avg_accuracy' in self.results[combination_key]:
                    combinations.append(f"{text_type}-{label_type}")
                    accuracies.append(self.results[combination_key].get('avg_accuracy', np.nan))
                    aucs.append(self.results[combination_key].get('avg_auc', np.nan))
                    f1_scores.append(self.results[combination_key].get('avg_f1_score', np.nan))
                    precisions.append(self.results[combination_key].get('avg_precision', np.nan))
                    recalls.append(self.results[combination_key].get('avg_recall', np.nan))
                    
                    # Add window size and focal alpha if available
                    window_sizes.append(self.results[combination_key].get('window_size', "N/A"))
                    focal_alphas.append(self.results[combination_key].get('focal_alpha', "N/A"))
        
        if not combinations:
            return
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'Combination': combinations,
            'Accuracy': accuracies,
            'AUC': aucs,
            'F1 Score': f1_scores,
            'Precision': precisions,
            'Recall': recalls,
            'Window Size': window_sizes,
            'Focal Alpha': focal_alphas
        })
        
        # Create a multi-metric visualization
        plt.figure(figsize=(15, 12))
        
        # Plot Accuracy
        plt.subplot(3, 2, 1)
        ax = sns.barplot(x='Combination', y='Accuracy', data=df)
        for i, p in enumerate(ax.patches):
            ax.annotate(f"W: {df['Window Size'][i]}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'bottom',
                       xytext = (0, 5), textcoords = 'offset points')
        plt.title('Accuracy by Model Combination - Fino1 Semantic Vectors')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot AUC
        plt.subplot(3, 2, 2)
        ax = sns.barplot(x='Combination', y='AUC', data=df)
        for i, p in enumerate(ax.patches):
            ax.annotate(f"W: {df['Window Size'][i]}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'bottom',
                       xytext = (0, 5), textcoords = 'offset points')
        plt.title('AUC by Model Combination - Fino1 Semantic Vectors')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot F1 Score
        plt.subplot(3, 2, 3)
        ax = sns.barplot(x='Combination', y='F1 Score', data=df)
        for i, p in enumerate(ax.patches):
            ax.annotate(f"α: {df['Focal Alpha'][i]}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'bottom',
                       xytext = (0, 5), textcoords = 'offset points')
        plt.title('F1 Score by Model Combination - Fino1 Semantic Vectors')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot Precision
        plt.subplot(3, 2, 4)
        ax = sns.barplot(x='Combination', y='Precision', data=df)
        plt.title('Precision by Model Combination - Fino1 Semantic Vectors')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot Recall
        plt.subplot(3, 2, 5)
        ax = sns.barplot(x='Combination', y='Recall', data=df)
        plt.title('Recall by Model Combination - Fino1 Semantic Vectors')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add a text summary
        plt.subplot(3, 2, 6)
        plt.axis('off')
        summary_text = "Performance Summary with Focal Loss:\n\n"
        
        for i, combo in enumerate(combinations):
            summary_text += f"{combo}:\n"
            summary_text += f"  Window Size: {window_sizes[i]}\n"
            summary_text += f"  Focal Alpha: {focal_alphas[i]}\n"
            summary_text += f"  Accuracy: {accuracies[i]:.4f}\n"
            summary_text += f"  F1 Score: {f1_scores[i]:.4f}\n"
            summary_text += f"  AUC: {aucs[i]:.4f}\n\n"
        
        plt.text(0, 0.5, summary_text, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('Climate_news_second_database/Fino1_semantic_LSTM_Plots/visualizations/COP/performance_comparisons/metrics_Fino1_SemanticVectors_focal.png')
        plt.close()
    
    def _create_performance_by_layer_visualizations(self):
        """Create visualizations comparing performance across layers."""
        # Collect data for Fino1 semantic vectors
        layer_data = []
        
        for text_type in ['Title', 'Full_text']:
            for label_type in ['Short-term', 'Long-term']:
                combination_key = f"Fino1_{text_type}_{label_type}"
                
                if combination_key in self.results:
                    combination = f"{text_type}-{label_type}"
                    window_size = self.results[combination_key].get('window_size', 'N/A')
                    focal_alpha = self.results[combination_key].get('focal_alpha', 'N/A')
                    
                    for i, layer_result in enumerate(self.results[combination_key].get('layer_results', [])):
                        layer_data.append({
                            'Combination': combination,
                            'Layer': f'Layer {i+1}',
                            'Accuracy': layer_result.get('accuracy', np.nan),
                            'AUC': layer_result.get('auc', np.nan),
                            'F1 Score': layer_result.get('f1_score', np.nan),
                            'Precision': layer_result.get('precision', np.nan),
                            'Recall': layer_result.get('recall', np.nan),
                            'Window Size': layer_result.get('window_size', window_size),
                            'Focal Alpha': layer_result.get('focal_alpha', focal_alpha)
                        })
        
        if not layer_data:
            return
        
        # Create dataframe for plotting
        df = pd.DataFrame(layer_data)
        
        # Create a multi-metric layer comparison visualization
        plt.figure(figsize=(18, 15))
        
        # Plot Accuracy by layer
        plt.subplot(3, 2, 1)
        sns.barplot(x='Layer', y='Accuracy', hue='Combination', data=df)
        plt.title('Accuracy by Layer and Combination - Fino1 Semantic Vectors')
        plt.ylim(0, 1.0)
        plt.legend(title='Combination', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot AUC by layer
        plt.subplot(3, 2, 2)
        sns.barplot(x='Layer', y='AUC', hue='Combination', data=df)
        plt.title('AUC by Layer and Combination - Fino1 Semantic Vectors')
        plt.ylim(0, 1.0)
        plt.legend(title='Combination', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot F1 Score by layer
        plt.subplot(3, 2, 3)
        sns.barplot(x='Layer', y='F1 Score', hue='Combination', data=df)
        plt.title('F1 Score by Layer and Combination - Fino1 Semantic Vectors')
        plt.ylim(0, 1.0)
        plt.legend(title='Combination', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot Precision by layer
        plt.subplot(3, 2, 4)
        sns.barplot(x='Layer', y='Precision', hue='Combination', data=df)
        plt.title('Precision by Layer and Combination - Fino1 Semantic Vectors')
        plt.ylim(0, 1.0)
        plt.legend(title='Combination', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot Recall by layer
        plt.subplot(3, 2, 5)
        sns.barplot(x='Layer', y='Recall', hue='Combination', data=df)
        plt.title('Recall by Layer and Combination - Fino1 Semantic Vectors')
        plt.ylim(0, 1.0)
        plt.legend(title='Combination', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('Climate_news_second_database/Fino1_semantic_LSTM_Plots/visualizations/COP/layer_comparisons/metrics_by_layer_Fino1_SemanticVectors_focal.png')
        plt.close()
    
    def print_summary(self):
        """Print a summary of all results."""
        print("\n" + "="*80)
        print("SUMMARY OF RESULTS FOR FINO1 SEMANTIC VECTORS WITH FOCAL LOSS (COP)")
        print("="*80)
        
        for text_type in ['Title', 'Full_text']:
            for label_type in ['Short-term', 'Long-term']:
                combination_key = f"Fino1_{text_type}_{label_type}"
                
                if combination_key in self.results and 'avg_accuracy' in self.results[combination_key]:
                    avg_accuracy = self.results[combination_key].get('avg_accuracy', np.nan)
                    avg_auc = self.results[combination_key].get('avg_auc', np.nan)
                    avg_f1 = self.results[combination_key].get('avg_f1_score', np.nan)
                    avg_precision = self.results[combination_key].get('avg_precision', np.nan)
                    avg_recall = self.results[combination_key].get('avg_recall', np.nan)
                    
                    window_size = self.results[combination_key].get('window_size', 'N/A')
                    focal_alpha = self.results[combination_key].get('focal_alpha', 'N/A')
                    focal_gamma = self.results[combination_key].get('focal_gamma', 'N/A')
                    
                    print(f"{text_type} + {label_type} (Window Size: {window_size}, Focal α: {focal_alpha}, γ: {focal_gamma}):")
                    print(f"  Average Accuracy: {avg_accuracy:.4f}")
                    print(f"  Average AUC: {avg_auc:.4f}")
                    print(f"  Average F1 Score: {avg_f1:.4f}")
                    print(f"  Average Precision: {avg_precision:.4f}")
                    print(f"  Average Recall: {avg_recall:.4f}")
                    
                    # Print layer-specific results
                    for i, layer_result in enumerate(self.results[combination_key].get('layer_results', [])):
                        accuracy = layer_result.get('accuracy', np.nan)
                        auc = layer_result.get('auc', np.nan)
                        f1 = layer_result.get('f1_score', np.nan)
                        precision = layer_result.get('precision', np.nan)
                        recall = layer_result.get('recall', np.nan)
                        
                        # Get confusion matrix stats if available
                        cm_stats = ""
                        if 'confusion_matrix' in layer_result:
                            cm = layer_result['confusion_matrix']
                            if cm.size == 4:  # 2x2 confusion matrix
                                tn, fp, fn, tp = cm.ravel()
                                cm_stats = f", TN={tn}, FP={fp}, FN={fn}, TP={tp}"
                                
                        print(f"    Layer {i+1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}{cm_stats}")
                        print(f"      Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            print()
        
        # Find best overall combination
        best_accuracy = 0
        best_auc = 0
        best_f1 = 0
        best_accuracy_combo = None
        best_auc_combo = None
        best_f1_combo = None
        
        for combo, results in self.results.items():
            if 'avg_accuracy' in results and not np.isnan(results['avg_accuracy']):
                if results['avg_accuracy'] > best_accuracy:
                    best_accuracy = results['avg_accuracy']
                    best_accuracy_combo = combo
                
                if 'avg_auc' in results and not np.isnan(results['avg_auc']) and results['avg_auc'] > best_auc:
                    best_auc = results['avg_auc']
                    best_auc_combo = combo
                    
                if 'avg_f1_score' in results and not np.isnan(results['avg_f1_score']) and results['avg_f1_score'] > best_f1:
                    best_f1 = results['avg_f1_score']
                    best_f1_combo = combo
        
        print("\nBest Overall Combinations:")
        print(f"Best Accuracy: {best_accuracy:.4f} - {best_accuracy_combo}")
        print(f"Best AUC: {best_auc:.4f} - {best_auc_combo}")
        print(f"Best F1 Score: {best_f1:.4f} - {best_f1_combo}")
        
        return self

# Main execution
if __name__ == "__main__":
    import sys
    import os
    
    # Check if we're running in a Jupyter environment
    is_jupyter = any(arg.endswith('json') for arg in sys.argv) or 'ipykernel' in sys.modules
    
    if is_jupyter:
        print("Running in Jupyter environment - using default settings")
        perform_grid_search = True  # Set your default here
        focal_gamma = 2.0
    else:
        # Only use argparse in non-Jupyter environments
        import argparse
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Stock trend prediction using Fino1 semantic vectors with focal loss')
        parser.add_argument('--grid-search', dest='grid_search', action='store_true',
                          help='Perform grid search for window size optimization')
        parser.add_argument('--no-grid-search', dest='grid_search', action='store_false',
                          help='Skip grid search and use default window size')
        parser.add_argument('--gamma', type=float, default=2.0,
                          help='Gamma parameter for focal loss (default: 2.0)')
        parser.set_defaults(grid_search=True)
        
        args = parser.parse_args()
        perform_grid_search = args.grid_search
        focal_gamma = args.gamma
        
    print(f"Running with grid search: {perform_grid_search}")
    print(f"Using focal loss with gamma: {focal_gamma}")
    
    # Initialize predictor with the new CSV file
    predictor = Fino1SemanticVectorPredictor('/home/c.c24004260/Climate_news_second_database/us_news_semantics_COP_completed_vectors_Fino1.csv')
    
    # Run prediction pipeline with focal loss
    predictor.load_data().define_time_windows().run_all_combinations(
        perform_grid_search=perform_grid_search,
        focal_gamma=focal_gamma
    )
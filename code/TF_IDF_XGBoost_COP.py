import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import xgboost as xgb
import warnings
import os
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

# Define climate-finance specific stopwords to remove in addition to regular stopwords
FINANCIAL_STOPWORDS = {
    'said', 'inc', 'corp', 'company', 'companies', 'reuters', 'news', 'press', 'release',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 
    'october', 'november', 'december', 'wall', 'street', 'journal'
}

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

class ImprovedClimateNewsPredictor:
    def __init__(self, csv_path):
        """
        Initialize the stock predictor for climate change news analysis with XGBoost,
        featuring optimized parameter spaces for different text inputs with anti-overfitting measures.
        
        Args:
            csv_path: Path to the CSV file containing climate change news data
        """
        self.csv_path = csv_path
        self.data = None
        self.layers = []
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english')).union(FINANCIAL_STOPWORDS)
        self.results = {}
        
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
        
        print(f"Loaded {len(self.data)} climate change news articles from Wall Street Journal spanning from "
              f"{self.data['Publication date'].min().strftime('%d/%m/%Y')} "
              f"to {self.data['Publication date'].max().strftime('%d/%m/%Y')}")
        print(f"Class distribution for short-term prediction: {self.data['S_label'].value_counts().to_dict()}")
        print(f"Class distribution for long-term prediction: {self.data['L_label'].value_counts().to_dict()}")
        
        return self
    
    def preprocess_text(self, text):
        """
        Preprocess climate change news text by applying various NLP techniques.
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Standardize climate-finance terms
        for term, replacement in CLIMATE_FINANCIAL_TERMS.items():
            text = re.sub(r'\b' + term + r'\b', replacement, text)
        
        # Standardize numerical representations with units
        text = re.sub(r'\$(\d+)([kmbt])', lambda m: m.group(1) + '_' + 
                      {'k': 'thousand', 'm': 'million', 'b': 'billion', 't': 'trillion'}[m.group(2).lower()], text)
        
        # Remove punctuation except $ and %
        text = re.sub(r'[^\w\s$%]', ' ', text)
        
        # Remove non-alphanumeric characters but preserve meaningful financial symbols
        text = re.sub(r'[^\w\s$%.-]', ' ', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stopwords]
        
        # Join tokens back into a string
        return ' '.join(tokens)
    
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
    
    def split_data(self, layer, text_col, label_col):
        """
        Split the data into training, validation, and test sets based on the defined time windows.
        
        Args:
            layer: The time window layer
            text_col: The column containing the text data ('Title' or 'Full text')
            label_col: The column containing the labels ('S_label' or 'L_label')
            
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
        
        # Preprocess text
        X_train = train_data[text_col].apply(self.preprocess_text)
        X_val = val_data[text_col].apply(self.preprocess_text)
        X_test = test_data[text_col].apply(self.preprocess_text)
        
        y_train = train_data[label_col]
        y_val = val_data[label_col]
        y_test = test_data[label_col]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), train_data, val_data, test_data
    
    def get_tfidf_parameters(self, text_col):
        """
        Get TF-IDF parameters based on text column with changes to reduce feature space for full text.
        
        Args:
            text_col: The text column ('Title' or 'Full text')
            
        Returns:
            Dictionary of TF-IDF parameters
        """
        if text_col == 'Title':
            return {
                'max_features': 1500,   # Reduced from 2000 to further constrain
                'min_df': 3,            # Keep as is - minimum document frequency
                'max_df': 0.85,         # Slightly reduced from 0.9 to filter more common terms
                'ngram_range': (1, 2),  # Keep unigrams and bigrams
                'sublinear_tf': True    # Apply sublinear scaling
            }
        else:  # 'Full text'
            return {
                'max_features': 4000,   # Reduced from 5000 to control complexity
                'min_df': 7,            # Increased from 5 to be more selective
                'max_df': 0.75,          # Reduced from 0.85 to filter more common terms
                'ngram_range': (1, 3),  # Reduced from (1,3) to limit feature explosion
                'sublinear_tf': True    # Apply sublinear scaling
            }
    
    def create_model(self, text_col):
        """
        Create a TF-IDF + XGBoost pipeline model with anti-overfitting measures.
        
        Args:
            text_col: The text column ('Title' or 'Full text')
        """
        # Get TF-IDF parameters based on text column
        tfidf_params = self.get_tfidf_parameters(text_col)
        
        # Create the pipeline with TF-IDF and XGBoost
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                random_state=42
            ))
        ])
        
        # Define parameters for the first stage grid search based on text column
        if text_col == 'Title':
            # Title-specific parameters (simpler models with stronger regularization)
            param_grid = {
                'classifier__n_estimators': [50, 100],            # Fewer trees
                'classifier__max_depth': [2, 3],                  # Reduced max depth (was 2-4)
                'classifier__learning_rate': [0.05, 0.1],         # Slightly lower learning rates
                'classifier__min_child_weight': [2, 3],           # Increased from 1 to prevent overfitting
                'classifier__subsample': [0.8, 0.9],              # Increased options for row sampling
                'classifier__colsample_bytree': [0.8, 0.9]        # Keep column sampling options
            }
        else:  # 'Full text'
            # Full text-specific parameters with STRONG anti-overfitting measures
            param_grid = {
                'classifier__n_estimators': [50, 80],             # Reduced from [100, 200, 300]
                'classifier__max_depth': [2, 3],                  # Significantly reduced from [3, 5, 7]
                'classifier__learning_rate': [0.01, 0.03, 0.05],  # Lower learning rates
                'classifier__min_child_weight': [3, 5],           # Increased to prevent overfitting
                'classifier__subsample': [0.6, 0.7],              # More aggressive row subsampling
                'classifier__colsample_bytree': [0.5, 0.6]        # More aggressive feature subsampling
            }
        
        # Create grid search model
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,                # 3-fold cross-validation
            scoring='roc_auc',   # Optimize for ROC AUC
            verbose=1,
            n_jobs=-1            # Use all available cores
        )
        
        return grid_search
    
    def create_second_stage_model(self, text_col, best_params):
        """
        Create a second stage model to fine-tune regularization parameters with anti-overfitting focus.
        
        Args:
            text_col: The text column ('Title' or 'Full text')
            best_params: Best parameters from the first stage
        """
        # Extract best core parameters from first stage
        n_estimators = best_params.get('classifier__n_estimators', 100)
        max_depth = best_params.get('classifier__max_depth', 3)
        learning_rate = best_params.get('classifier__learning_rate', 0.05)
        min_child_weight = best_params.get('classifier__min_child_weight', 3)
        subsample = best_params.get('classifier__subsample', 0.8)
        colsample_bytree = best_params.get('classifier__colsample_bytree', 0.8)
        
        # Get TF-IDF parameters based on text column
        tfidf_params = self.get_tfidf_parameters(text_col)
        
        # Create the pipeline with TF-IDF and XGBoost
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                random_state=42
            ))
        ])
        
        # Define parameters for the second stage grid search based on text column
        if text_col == 'Title':
            # Title-specific parameters for second stage (focus on regularization)
            param_grid = {
                'classifier__gamma': [0, 0.1, 0.2],           # Minimum loss reduction for splitting
                'classifier__reg_alpha': [0.1, 0.5, 1.0],     # L1 regularization
                'classifier__reg_lambda': [1.0, 2.0, 3.0]     # L2 regularization
            }
        else:  # 'Full text'
            # Full text-specific parameters for second stage with stronger regularization
            param_grid = {
                'classifier__gamma': [0.1, 0.3, 0.5],         # Increased minimum loss reduction
                'classifier__reg_alpha': [0.5, 1.0, 2.0],     # Stronger L1 regularization
                'classifier__reg_lambda': [2.0, 3.0, 5.0]     # Stronger L2 regularization
            }
        
        # Create grid search model
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            scoring='roc_auc',
            verbose=1,
            n_jobs=-1
        )
        
        return grid_search
    
    def train_and_evaluate(self, text_col, label_col):
        """
        Train and evaluate the model for a specific text column and label column.
        
        Args:
            text_col: The text column to use ('Title' or 'Full text')
            label_col: The label column to use ('S_label' or 'L_label')
        """
        # Store results
        combination_key = f"{text_col}|{label_col}"
        self.results[combination_key] = {
            'accuracy': [],
            'auc': [],
            'best_params_stage1': [],
            'best_params_stage2': [],
            'layer_results': []
        }
        
        print(f"\n{'='*80}")
        print(f"Training XGBoost model for {text_col} and {label_col}")
        print(f"{'='*80}")
        
        for i, layer in enumerate(self.layers):
            print(f"\nLayer {i+1}:")
            print(f"Training period: {layer['train_start'].strftime('%d/%m/%Y')} - {layer['train_end'].strftime('%d/%m/%Y')}")
            print(f"Validation period: {layer['val_start'].strftime('%d/%m/%Y')} - {layer['val_end'].strftime('%d/%m/%Y')}")
            print(f"Testing period: {layer['test_start'].strftime('%d/%m/%Y')} - {layer['test_end'].strftime('%d/%m/%Y')}")
            
            # Split data
            (X_train, y_train), (X_val, y_val), (X_test, y_test), train_data, val_data, test_data = self.split_data(layer, text_col, label_col)
            
            # Check if there are enough samples and classes
            if len(X_train) < 10 or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
                print(f"Skipping layer {i+1} due to insufficient data or class imbalance")
                continue
            
            # Stage 1: Create and train model with core parameters optimized for input type
            print(f"Stage 1: Tuning core parameters for {text_col}...")
            model_stage1 = self.create_model(text_col)
            model_stage1.fit(X_train, y_train)
            
            # Get best model and parameters from stage 1
            best_model_stage1 = model_stage1.best_estimator_
            best_params_stage1 = model_stage1.best_params_
            
            print(f"Best parameters from Stage 1: {best_params_stage1}")
            self.results[combination_key]['best_params_stage1'].append(best_params_stage1)
            
            # Stage 2: Fine-tune regularization parameters
            print(f"Stage 2: Fine-tuning regularization parameters for {text_col}...")
            model_stage2 = self.create_second_stage_model(text_col, best_params_stage1)
            model_stage2.fit(X_train, y_train)
            
            # Get best model and parameters from stage 2
            best_model = model_stage2.best_estimator_
            best_params_stage2 = model_stage2.best_params_
            
            print(f"Best parameters from Stage 2: {best_params_stage2}")
            self.results[combination_key]['best_params_stage2'].append(best_params_stage2)
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probability of class 1
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            self.results[combination_key]['accuracy'].append(accuracy)
            self.results[combination_key]['auc'].append(auc)
            
            print(f"Test Accuracy for Layer {i+1}: {accuracy:.4f}")
            print(f"Test AUC for Layer {i+1}: {auc:.4f}")
            
            # Store layer results for visualization
            layer_result = {
                'layer': i+1,
                'model': best_model,
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy,
                'auc': auc,
                'best_params_stage1': best_params_stage1,
                'best_params_stage2': best_params_stage2
            }
            
            self.results[combination_key]['layer_results'].append(layer_result)
            
            # Create learning curve visualization
            self.plot_learning_curves(layer_result, text_col, label_col, i+1)
        
        # Calculate average metrics
        avg_accuracy = np.mean(self.results[combination_key]['accuracy']) if self.results[combination_key]['accuracy'] else 0
        avg_auc = np.mean(self.results[combination_key]['auc']) if self.results[combination_key]['auc'] else 0
        
        print(f"\nAverage Test Accuracy across all layers: {avg_accuracy:.4f}")
        print(f"Average Test AUC across all layers: {avg_auc:.4f}")
        
        self.results[combination_key]['avg_accuracy'] = avg_accuracy
        self.results[combination_key]['avg_auc'] = avg_auc
        
        return self
    
    def plot_learning_curves(self, layer_result, text_col, label_col, layer_num):
        """Plot learning curves to detect overfitting with improved styling."""
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Learning Curves: {text_col} + {label_col} (Layer {layer_num})", fontsize=16)
        
        X_train = layer_result['X_train']
        y_train = layer_result['y_train']
        
        # Create different training set sizes
        train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
        train_acc = []
        val_acc = []
        #train_auc = []
        #val_auc = []
        
        for size in train_sizes:
            # Get subset of training data
            n_samples = int(len(X_train) * size)
            X_train_subset = X_train.iloc[:n_samples]
            y_train_subset = y_train.iloc[:n_samples]
            
            # Train a new model
            model_clone = clone(layer_result['model'])
            model_clone.fit(X_train_subset, y_train_subset)
            
            # Evaluate on training and validation sets
            y_train_pred = model_clone.predict(X_train_subset)
            y_val_pred = model_clone.predict(layer_result['X_val'])
            
            #y_train_proba = model_clone.predict_proba(X_train_subset)[:, 1]
            #y_val_proba = model_clone.predict_proba(layer_result['X_val'])[:, 1]
            
            train_acc.append(accuracy_score(y_train_subset, y_train_pred))
            val_acc.append(accuracy_score(layer_result['y_val'], y_val_pred))
            
            #train_auc.append(roc_auc_score(y_train_subset, y_train_proba))
            #val_auc.append(roc_auc_score(layer_result['y_val'], y_val_proba))
        
        # Plot with improved styling (no grid lines)
        #plt.subplot(1, 2, 1)
        plt.plot(train_sizes, train_acc, 'o-', label='Training Accuracy', color='blue')
        plt.plot(train_sizes, val_acc, 'o-', label='Validation Accuracy', color='orange')
        plt.title('Accuracy Learning Curve')
        plt.xlabel('Training Set Size (%)')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        
        #plt.subplot(1, 2, 2)
        #plt.plot(train_sizes, train_auc, 'o-', label='Training AUC', color='blue')
        #plt.plot(train_sizes, val_auc, 'o-', label='Validation AUC', color='orange')
        #plt.title('AUC Learning Curve')
        #plt.xlabel('Training Set Size (%)')
        #plt.ylabel('AUC')
        #plt.legend(loc='lower right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Create directory for visualizations if it doesn't exist
        viz_dir = f'Climate_news_second_database/TF_IDF_XGBoost/visualizations_{text_col.lower().replace(" ", "_")}/COP'
        os.makedirs(viz_dir, exist_ok=True)
        save_path = os.path.join(viz_dir, f"learning_curves_{text_col}_{label_col}_layer_{layer_num}.png")
        plt.savefig(save_path)
        print(f"Learning curves visualization saved as: {save_path}")
        plt.close()
    
    def run_all_combinations(self):
        """Run the analysis for all combinations of text and label columns."""
        # Define all combinations
        combinations = [
            ('Title', 'S_label'),     # news headline + short-term prediction
            ('Title', 'L_label'),     # news headline + long-term prediction
            ('Full text', 'S_label'), # news body + short-term prediction
            ('Full text', 'L_label')  # news body + long-term prediction
        ]
        
        # Run analysis for each combination
        for text_col, label_col in combinations:
            self.train_and_evaluate(text_col, label_col)
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY OF RESULTS (COP)")
        print("="*80)
        
        # Create a summary table for easier comparison
        summary_data = []
        
        for combination, results in self.results.items():
            text_col, label_col = combination.split('|')
            avg_accuracy = results.get('avg_accuracy', 'N/A')
            avg_auc = results.get('avg_auc', 'N/A')
            
            print(f"\nCombination: {text_col} + {label_col}")
            print(f"Average Accuracy: {avg_accuracy:.4f}")
            print(f"Average AUC: {avg_auc:.4f}")
            
            # Print layer-specific results
            for i, accuracy in enumerate(results.get('accuracy', [])):
                auc = results.get('auc', [])[i]
                print(f"  Layer {i+1} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
            summary_data.append({
                'Combination': f"{text_col} + {label_col}",
                'Avg Accuracy': f"{avg_accuracy:.4f}",
                'Avg AUC': f"{avg_auc:.4f}"
            })
        
        # Create summary visualization
        self.create_summary_visualization(summary_data)
        
        return self
    
    def create_summary_visualization(self, summary_data):
        """Create a summary visualization comparing all model combinations with improved styling."""
        if not summary_data:
            print("No data available for summary visualization")
            return
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame(summary_data)
        
        # Convert string metrics to float for plotting
        df['Avg Accuracy'] = df['Avg Accuracy'].astype(float)
        df['Avg AUC'] = df['Avg AUC'].astype(float)
        
        # Create bar chart comparing all combinations with improved styling
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy bars
        x = np.arange(len(df))
        width = 0.35
        
        plt.bar(x - width/2, df['Avg Accuracy'], width, label='Average Accuracy', color='skyblue')
        plt.bar(x + width/2, df['Avg AUC'], width, label='Average AUC', color='salmon')
        
        plt.xlabel('Model Combination')
        plt.ylabel('Score')
        plt.title('Performance Comparison of Different Model Combinations')
        plt.xticks(x, df['Combination'], rotation=0, ha='center')
        plt.legend()
        
        # Add value labels on top of bars
        for i, v in enumerate(df['Avg Accuracy']):
            plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
        
        for i, v in enumerate(df['Avg AUC']):
            plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
        
        plt.ylim(0, max(df['Avg Accuracy'].max(), df['Avg AUC'].max()) + 0.1)
        plt.tight_layout()
        
        # Create visualizations directory if it doesn't exist
        os.makedirs('Climate_news_second_database/TF_IDF_XGBoost/visualizations_summary/COP', exist_ok=True)
        save_path = os.path.join('Climate_news_second_database/TF_IDF_XGBoost/visualizations_summary/COP', "performance_comparison.png")
        plt.savefig(save_path)
        print(f"Summary comparison visualization saved as: {save_path}")
        plt.close()
        
        # Create a more detailed visualization showing layer-by-layer performance
        self.create_layer_performance_visualization()
    
    def create_layer_performance_visualization(self):
        """Create visualization comparing performance across different layers with improved styling."""
        # Prepare data for visualization
        layer_data = []
        
        for combination, results in self.results.items():
            text_col, label_col = combination.split('|')
            for i, accuracy in enumerate(results.get('accuracy', [])):
                auc = results.get('auc', [])[i]
                layer_data.append({
                    'Combination': f"{text_col} + {label_col}",
                    'Layer': f"Layer {i+1}",
                    'Accuracy': accuracy,
                    'AUC': auc
                })
        
        # Skip if no data is available
        if not layer_data:
            return
        
        # Create DataFrame
        df = pd.DataFrame(layer_data)
        
        # Create visualization with improved styling
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        
        # Plot Accuracy by layer
        sns.barplot(x='Combination', y='Accuracy', hue='Layer', data=df, ax=ax1)
        ax1.set_title('Accuracy by Model Combination and Layer')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 0.8)  # Adjusted y-axis range as requested
        
        # Plot AUC by layer
        sns.barplot(x='Combination', y='AUC', hue='Layer', data=df, ax=ax2)
        ax2.set_title('AUC by Model Combination and Layer')
        ax2.set_ylabel('AUC')
        ax2.set_ylim(0, 0.8)  # Adjusted y-axis range as requested
        
        # Adjust layout
        plt.tight_layout()
        plt.xticks(rotation=0)  # Horizontal labels
        
        # Save the visualization
        save_path = os.path.join('Climate_news_second_database/TF_IDF_XGBoost/visualizations_summary/COP', "layer_performance_comparison.png")
        plt.savefig(save_path)
        print(f"Layer performance visualization saved as: {save_path}")
        plt.close()


# Main execution
if __name__ == "__main__":
    # Ensure all visualization directories exist
    for directory in ['Climate_news_second_database/TF_IDF_XGBoost/visualizations_title/COP', 'Climate_news_second_database/TF_IDF_XGBoost/visualizations_full_text/COP', 'Climate_news_second_database/TF_IDF_XGBoost/visualizations_summary']:
        os.makedirs(directory, exist_ok=True)
    
    predictor = ImprovedClimateNewsPredictor('/home/c.c24004260/Climate_news_second_database/us_news_semantics_COP_completed.csv')
    predictor.load_data().define_time_windows().run_all_combinations()
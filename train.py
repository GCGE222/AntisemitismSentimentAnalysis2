#RUN ON WINDOWS AND GPU SUPPORT MAAY NOT WORK FOR NVIDA. IF CUDA DOES NOT WORK IT WILL USE CPU TO RUN MODEL
'''
Note that this code was written before the training dataset included non-twitterdata and was then modified to fit the training dataset,
This is why some comments and function/class names use the label of 'tweet' instead of something else
'''
import pandas as pd
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    get_linear_schedule_with_warmup
)
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import nltk
from nltk.corpus import stopwords
import logging
from tqdm.auto import tqdm
import argparse
import os
import platform
import warnings
import numpy as np

# Configure logging and warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data and set Stop_Words
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

#Tracks training metrics
class MetricsTracker:
    #Initializes
    def __init__(self):
        #create array for metric history
        self.metrics_history = []
        #initialize current epoch at zero
        self.current_epoch = 0
        #create directory for storing metrics
        self.metrics_dir = 'training_metrics'
        #create directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.master_file = os.path.join(self.metrics_dir, 'all_metrics.csv')
    
    #Aggregate metric results while handeling Binary Classification
    def compute_aggregate_metrics(self, labels, predictions, val_loss):
        # Handle error if the Bias prediction is not binary
        unique_labels = np.unique(labels)
        unique_preds = np.unique(predictions)
        if len(unique_preds) == 1 and len(unique_labels) > 1:
            logger.warning("Model is predicting only one class. Check class balance and model training.")
        
        # Calculate metrics with proper binary averaging
        precision = precision_score(labels, predictions, average='binary', zero_division=0)
        recall = recall_score(labels, predictions, average='binary', zero_division=0)
        f1 = f1_score(labels, predictions, average='binary', zero_division=0)
        accuracy = accuracy_score(labels, predictions)
        
        # Calculate ratios for predictions in order to identify overtraining
        positive_ratio = np.mean(labels)
        pred_positive_ratio = np.mean(predictions)
        
        metrics = {
            'epoch': self.current_epoch,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'validation_loss': val_loss,
            'true_positive_ratio': positive_ratio,
            'predicted_positive_ratio': pred_positive_ratio
        }
        
        return metrics
    
    def compute_tweet_metrics(self, texts, labels, predictions, scores):
        #Compute detailed metrics for each Text entry from the Dataset
        tweet_metrics = []
        
        for text, label, pred, score in zip(texts, labels, predictions, scores):
            tweet_data = {
                'text': text,
                'true_label': int(label),
                'predicted_label': int(pred),
                'prediction_score': float(score),
                'correct_prediction': int(label == pred),
                'epoch': self.current_epoch,
                'confidence': float(max(score, 1-score)),
                'error_type': self._get_error_type(label, pred)
            }
            tweet_metrics.append(tweet_data)
            
        return pd.DataFrame(tweet_metrics)
    
    def _get_error_type(self, true_label, pred_label):
        """Categorize prediction errors"""
        if true_label == pred_label:
            return 'correct'
        elif true_label == 1:
            return 'false_negative'
        else:
            return 'false_positive'
    
    def update_metrics(self, texts, labels, predictions, scores, val_loss):
        """Update metrics and save to files"""
        # Compute all metrics
        agg_metrics = self.compute_aggregate_metrics(labels, predictions, val_loss)
        self.metrics_history.append(agg_metrics)
        
        # Compute and save tweet-level metrics
        tweet_metrics_df = self.compute_tweet_metrics(texts, labels, predictions, scores)
        tweet_metrics_file = os.path.join(self.metrics_dir, f'epoch_{self.current_epoch}_tweet_metrics.csv')
        tweet_metrics_df.to_csv(tweet_metrics_file, index=False)
        
        # Update master metrics file
        pd.DataFrame(self.metrics_history).to_csv(self.master_file, index=False)
        
        # Log metric summaries
        logger.info(f"\nEpoch {self.current_epoch} Metrics:")
        logger.info(f"Validation Loss: {agg_metrics['validation_loss']:.4f}") #was giving me problems, remove if necessary
        logger.info(f"Accuracy: {agg_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {agg_metrics['precision']:.4f}")
        logger.info(f"Recall: {agg_metrics['recall']:.4f}")
        logger.info(f"F1 Score: {agg_metrics['f1_score']:.4f}")
        logger.info(f"True Positive Ratio: {agg_metrics['true_positive_ratio']:.4f}")
        logger.info(f"Predicted Positive Ratio: {agg_metrics['predicted_positive_ratio']:.4f}")
        
        self.current_epoch += 1
        return agg_metrics, tweet_metrics_df

#Handles Dataset logic
class TweetDataset(Dataset):
    #Handles Intialization of model
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'texts': text
        }

def setup_device():
    logger.info("Detecting available compute devices...")
    
    if platform.system() != 'Windows':
        logger.warning("This script is optimized for Windows. Some features might not work on other systems.")
    
    try:
        import torch_directml
        device = torch_directml.device()
        logger.info("Successfully initialized DirectML for AMD GPU acceleration")
        return device
    except ImportError:
        logger.warning("\nDirectML not found. Installing required package for AMD GPU support...")
        os.system('pip install torch-directml')
        try:
            import torch_directml
            device = torch_directml.device()
            logger.info("Successfully installed and initialized DirectML for AMD GPU")
            return device
        except Exception as e:
            logger.error(f"Failed to setup DirectML: {str(e)}")
            logger.info("Falling back to CPU...")
            return torch.device("cpu")

def move_to_device(batch, device):
    """Safely move batch to device with error handling"""
    try:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    except RuntimeError as e:
        logger.error(f"Error moving batch to device: {str(e)}")
        raise

#Data Cleaning
def preprocess_text(text):
    """Clean and preprocess text data"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = [word for word in text.split() if word not in STOP_WORDS]
    return ' '.join(words)

# loaad data input test size
def load_and_preprocess_data(file_path, test_size=0.3):
    """Load and preprocess the dataset"""
    logger.info("Loading and preprocessing data...")
    
    # Check if training file exists and load the dataset
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training file not found: {file_path}")
    #Read user training CSV file
    df = pd.read_csv(file_path)
    
    # Check if required columns exist in the dataset and raise an error if not found
    required_columns = ['Text', 'Biased', 'Keyword']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need: {required_columns}")
    
    # Clean data
    df = df[required_columns].copy()
    df = df.dropna(subset=['Text', 'Biased'])
    #Make all biased scores integers to fix the dataset merging inconsistency
    df['Biased'] = pd.to_numeric(df['Biased'], errors='coerce')
    df = df.dropna(subset=['Biased'])
    #apply the preprocessing
    df['Text'] = df['Text'].apply(preprocess_text)
    #strip the leading whitespace from text
    df = df[df['Text'].str.strip().str.len() > 0]
    
    # Split data
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['Biased'],
            random_state=42
        )
    except ValueError:
        logger.warning("Stratified split failed, using random split")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    logger.info(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    return train_df, test_df

def create_data_loaders(train_df, test_df, tokenizer, batch_size):
    # Create dataset objects from DataFrames
    train_dataset = TweetDataset(train_df['Text'].values, train_df['Biased'].values, tokenizer)
    test_dataset = TweetDataset(test_df['Text'].values, test_df['Biased'].values, tokenizer)
    
    # Handle class imbalance through weighted sampling
    labels = train_df['Biased'].values
    class_counts = np.bincount(labels.astype(int))  # Count instances of each class
    weights = 1. / class_counts  # Inverse frequency weighting
    sample_weights = weights[labels.astype(int)]  # Weight for each sample
    
    # Example: if class_counts = [80, 20] (80 negative, 20 positive samples)
    # weights would be [1/80, 1/20] = [0.0125, 0.05]
    # This gives higher sampling probability to minority class
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True  # Allows sampling same instance multiple times
    )
    # train smaple
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )
    #Test Sample
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, test_loader

#metric evaluation
def evaluate(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    prediction_scores = []
    texts = []
    total_loss = 0
    
    with torch.no_grad(): # Disable gradient calculation for efficiency
        for batch in tqdm(test_loader, desc="Evaluating"):
            try:
                batch_texts = batch.pop('texts')
                batch = move_to_device(batch, device)
                
                # Get model predictions
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)  # Convert to probabilities
                preds = torch.argmax(logits, dim=1)   #Get predicted class
                
                # Collect results
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())
                prediction_scores.extend(probs[:, 1].cpu().numpy())
                texts.extend(batch_texts)
                
            except RuntimeError as e:
                logger.error(f"Error during evaluation: {str(e)}")
                continue
    
    return {
        'loss': total_loss / len(test_loader),
        'predictions': predictions,
        'true_labels': true_labels,
        'scores': prediction_scores,
        'texts': texts
    }
#Saves Model
def save_model(model, tokenizer, save_path):
    
    logger.info("Saving model...")
    try:
        # Create a CPU copy of the model for saving
        cpu_model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2,
            state_dict={k: v.cpu() for k, v in model.state_dict().items()}
        )
        
        os.makedirs(save_path, exist_ok=True)
        cpu_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved successfully to {save_path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise
#Trains the Model
def train_model(train_df, test_df, epochs=3, batch_size=16, learning_rate=2e-5,):#Patience arg goes here    
    #sets up weather the training will be handled by the CPU or GPU
    device = setup_device()
    #Initialize mertic tracking
    metrics_tracker = MetricsTracker()
    
    # Initialize model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        #set model standards to best fit training data
        num_labels=2,
        dropout=0.2
    ).to(device)
    
    # Create the training and test data loaders with balanced sampling
    train_loader, test_loader = create_data_loaders(train_df, test_df, tokenizer, batch_size)
    
    # Set up the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    num_training_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')  # Initialize with infinity following Huggingface instructions
    patience = input(int('What patience score should this model use?'))
    no_improvement = 0
    
    #Training Loop
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        try:
            # Training phase
            model.train()
            total_train_loss = 0
            train_steps = 0
            
            # Training progress bar
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            for batch in progress_bar:
                #Try statment for error handeling
                try:
                    #set gradients to zero
                    optimizer.zero_grad(set_to_none=True)
                    #extract the actual text data from the batch of data that is being processed during the training and evaluation phases.
                    batch_texts = batch.pop('texts')
                    batch = move_to_device(batch, device)
                    
                    #Pass the model through and calculate loss            
                    outputs = model(**batch)
                    loss = outputs.loss
                    #Backpropagation and weight adjustment
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    # Update metrics
                    total_train_loss += loss.item()
                    train_steps += 1
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{total_train_loss/train_steps:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                   
                except RuntimeError as e:
                    logger.error(f"Error in training batch: {str(e)}")
                    continue
            
            #calculate training loss and logg metric
            avg_train_loss = total_train_loss / train_steps
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Evaluate
            eval_results = evaluate(model, test_loader, device)
            current_val_loss = eval_results['loss']
            
            # Update metrics
            metrics, tweet_metrics = metrics_tracker.update_metrics(
                texts=eval_results['texts'],
                labels=eval_results['true_labels'],
                predictions=eval_results['predictions'],
                scores=eval_results['scores'],
                val_loss=current_val_loss
            )
            
            # Save model with lowest validation loss
            if current_val_loss < best_val_loss:
                improvement = best_val_loss - current_val_loss
                best_val_loss = current_val_loss
                #no_improvement = 0
                logger.info(f"New best validation loss: {best_val_loss:.4f} (improved by {improvement:.4f})")
                #save best model and handle error
                try:
                    save_model(model, tokenizer, 'best_model')
                except Exception as e:
                    logger.error(f"Error saving model: {str(e)}")
            #if the model stops improving count patience to save resources
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    logger.info(f"Early stopping triggered. Best validation loss: {best_val_loss:.4f}")
                    break
        #handle error           
        except Exception as e:
            logger.error(f"Error in epoch {epoch + 1}: {str(e)}")
            continue
    
    logger.info("\nTraining completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, tokenizer, metrics_tracker.metrics_history
def main():
    #Use argument parsing for versatility
    parser = argparse.ArgumentParser(description='Train bias detection model (AMD GPU optimized)')
    parser.add_argument('file_path', help='Path to training dataset CSV')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    #parser.add_argument('--patience', type=int, default=3, help='Iterations of traininf if there is no improvement')
    
    args = parser.parse_args()
    
    try:
        # Print training configuration
        logger.info("\nTraining Configuration:")
        logger.info(f"Data file: {args.file_path}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Max sequence length: {args.max_len}")
        
        # Load and preprocess data
        train_df, test_df = load_and_preprocess_data(args.file_path)
        
        # Train model
        model, tokenizer, metrics_history = train_model(
            train_df,
            test_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            #patience=args.patience
        )
        
        # Save final metrics summary
        final_metrics_file = os.path.join('training_metrics', 'final_summary.txt')
        with open(final_metrics_file, 'w') as f:
            f.write("Training Summary\n")
            f.write("================\n\n")
            f.write(f"Total epochs trained: {len(metrics_history)}\n")
            f.write(f"Best validation loss: {min(m['validation_loss'] for m in metrics_history):.4f}\n")
            f.write(f"Final accuracy: {metrics_history[-1]['accuracy']:.4f}\n")
            f.write(f"Final precision: {metrics_history[-1]['precision']:.4f}\n")
            f.write(f"Final recall: {metrics_history[-1]['recall']:.4f}\n")
            f.write(f"Final F1 score: {metrics_history[-1]['f1_score']:.4f}\n")
            f.write(f"Final validation loss: {metrics_history[-1]['validation_loss']:.4f}\n")
            
            logger.info("Training completed successfully!")
            logger.info(f"Final metrics summary saved to {final_metrics_file}")
    #Error handling    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()
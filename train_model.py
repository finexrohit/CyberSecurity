"""
Cyberbullying Detection - Model Training Script

This script trains a machine learning model for cyberbullying detection
using text classification techniques.

Features:
- Data preprocessing with NLP techniques
- TF-IDF vectorization
- Multiple classifier comparison
- Model evaluation and saving

Author: College Project Team
Date: December 2025

Usage:
    python train_model.py
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# NLP Libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError:
    print("Installing NLTK...")
    os.system('pip install nltk')
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

# Machine Learning Libraries
try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, classification_report, confusion_matrix
    )
except ImportError:
    print("Installing scikit-learn...")
    os.system('pip install scikit-learn')
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, classification_report, confusion_matrix
    )

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)


def create_sample_dataset():
    """
    Create a sample dataset for training when no external dataset is available.
    This is a simplified dataset for demonstration purposes.
    
    Returns:
        pd.DataFrame: Sample dataset with 'text' and 'label' columns
    """
    print("\nCreating sample dataset for demonstration...")
    
    # Sample cyberbullying texts (label = 1)
    bullying_texts = [
        "You are so ugly and stupid, nobody likes you",
        "I hate you so much, just go away and die",
        "You're a worthless loser with no friends",
        "Everyone thinks you're pathetic and disgusting",
        "You should kill yourself, nobody would care",
        "You're so fat and ugly, it's disgusting",
        "I hope something bad happens to you today",
        "You're the dumbest person I've ever met",
        "Nobody wants you here, just leave forever",
        "You're a waste of space and oxygen",
        "Your face is so ugly it makes me sick",
        "I wish you were never born, you idiot",
        "You're nothing but a stupid worthless fool",
        "Everyone hates you and talks behind your back",
        "You deserve all the bad things happening to you",
        "You're so pathetic it's actually funny",
        "Go cry somewhere else you little baby",
        "You're a complete failure at everything",
        "Nobody will ever love someone as ugly as you",
        "You're the worst person I've ever known",
        "I can't believe how stupid you are",
        "You make me want to throw up",
        "Your existence is a mistake",
        "You're embarrassing yourself constantly",
        "Everyone laughs at you behind your back",
        "You're such an attention seeking loser",
        "Why don't you just disappear forever",
        "You're completely useless and talentless",
        "I hate everything about you",
        "You're a disgusting human being",
        "Nobody asked for your opinion idiot",
        "You should be ashamed of yourself",
        "You're the reason nobody likes our group",
        "Stop trying so hard, you're embarrassing",
        "You're so annoying everyone avoids you",
        "I can't stand looking at your face",
        "You're a total disappointment to everyone",
        "Your ideas are stupid like you",
        "Nobody cares about what you think",
        "You're such a fake loser"
    ]
    
    # Sample non-bullying texts (label = 0)
    non_bullying_texts = [
        "Great job on your presentation today!",
        "I really enjoyed working with you on this project",
        "Thank you for helping me with my homework",
        "You're such a kind and helpful person",
        "I appreciate your feedback on my work",
        "Let's meet up for coffee this weekend",
        "Happy birthday! Hope you have a wonderful day",
        "Congratulations on your achievement!",
        "Your hard work really paid off",
        "I'm proud of how much you've improved",
        "Thanks for always being there for me",
        "You did an amazing job on that test",
        "I love your creative ideas",
        "You're an inspiration to all of us",
        "Keep up the excellent work",
        "It was nice meeting you today",
        "I hope you feel better soon",
        "Your presentation was very informative",
        "I learned a lot from your explanation",
        "You have such a positive attitude",
        "Thanks for the great suggestions",
        "I enjoyed our conversation yesterday",
        "You're making great progress",
        "Your enthusiasm is contagious",
        "I'm grateful for your support",
        "You always know how to cheer me up",
        "That was a thoughtful thing to do",
        "I respect your opinion on this matter",
        "You handled that situation well",
        "I appreciate your patience with me",
        "Your kindness means a lot to me",
        "You have a talent for this",
        "I'm impressed by your dedication",
        "Thanks for understanding",
        "You're a great team player",
        "I value our friendship",
        "Your effort is noticeable",
        "That's a brilliant idea!",
        "I admire your determination",
        "You should be proud of yourself"
    ]
    
    # Sample neutral/potentially harmful texts (label = 2) - optional category
    neutral_texts = [
        "That was kind of dumb but okay",
        "I disagree with your point completely",
        "You could do better than this",
        "That's not how you should do it",
        "Your work needs a lot of improvement",
        "I don't think that's a good idea",
        "You're wrong about this",
        "That was a bad decision",
        "I expected more from you",
        "This is disappointing work",
        "You need to try harder next time",
        "That's not very smart",
        "I'm not impressed at all",
        "You should reconsider your approach",
        "This doesn't make sense",
        "I don't agree with anything you said",
        "That's a weak argument",
        "You missed the point entirely",
        "This could be much better",
        "I have concerns about your work"
    ]
    
    # Combine all texts
    texts = bullying_texts + non_bullying_texts + neutral_texts
    labels = [1] * len(bullying_texts) + [0] * len(non_bullying_texts) + [2] * len(neutral_texts)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Created dataset with {len(df)} samples:")
    print(f"  - Bullying: {len(bullying_texts)}")
    print(f"  - Non-Bullying: {len(non_bullying_texts)}")
    print(f"  - Neutral: {len(neutral_texts)}")
    
    return df


def preprocess_text(text):
    """
    Preprocess text using NLP techniques.
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return text


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple ML models.
    
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing labels
        
    Returns:
        dict: Results for each model
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(kernel='linear', probability=True, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results


def select_best_model(results):
    """
    Select the best performing model based on F1-score.
    
    Args:
        results (dict): Dictionary of model results
        
    Returns:
        tuple: (model_name, model, metrics)
    """
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_result = results[best_model_name]
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print("="*60)
    print(f"  Accuracy:  {best_result['accuracy']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print(f"  F1-Score:  {best_result['f1_score']:.4f}")
    
    return best_model_name, best_result['model'], best_result


def save_model(model, vectorizer, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    """
    Save the trained model and vectorizer to disk.
    
    Args:
        model: Trained ML model
        vectorizer: Fitted TF-IDF vectorizer
        model_path (str): Path to save the model
        vectorizer_path (str): Path to save the vectorizer
    """
    print("\nSaving model and vectorizer...")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved to: {model_path}")
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  Vectorizer saved to: {vectorizer_path}")


def main():
    """
    Main function to orchestrate the training pipeline.
    """
    print("\n" + "="*60)
    print("CYBERBULLYING DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load or create dataset
    # Check if external dataset exists
    dataset_paths = ['dataset.csv', 'data/cyberbullying_data.csv', 'cyberbullying_tweets.csv']
    df = None
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"\nLoading dataset from: {path}")
            df = pd.read_csv(path)
            break
    
    if df is None:
        df = create_sample_dataset()
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Ensure we have 'text' and 'label' columns
    if 'text' not in df.columns:
        # Try to find the text column
        text_cols = [col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower() or 'comment' in col.lower()]
        if text_cols:
            df = df.rename(columns={text_cols[0]: 'text'})
    
    if 'label' not in df.columns:
        # Try to find the label column
        label_cols = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower() or 'target' in col.lower()]
        if label_cols:
            df = df.rename(columns={label_cols[0]: 'label'})
    
    # Step 2: Preprocess text
    print("\nPreprocessing text data...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]
    print(f"Dataset after cleaning: {df.shape}")
    
    # Step 3: Split data
    print("\nSplitting dataset...")
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Testing set:  {len(X_test)} samples")
    
    # Step 4: Feature extraction using TF-IDF
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"  Feature matrix shape: {X_train_tfidf.shape}")
    
    # Step 5: Train and evaluate models
    results = train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test)
    
    # Step 6: Select best model
    best_name, best_model, best_metrics = select_best_model(results)
    
    # Step 7: Save model and vectorizer
    save_model(best_model, vectorizer)
    
    # Step 8: Test prediction
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    test_texts = [
        "You are so stupid and ugly!",
        "Great job on your presentation!",
        "I hate everything about you",
        "Thanks for helping me today"
    ]
    
    for text in test_texts:
        cleaned = preprocess_text(text)
        features = vectorizer.transform([cleaned])
        prediction = best_model.predict(features)[0]
        probability = best_model.predict_proba(features)[0]
        
        labels = {0: 'Non-Bullying', 1: 'Bullying', 2: 'Neutral'}
        
        print(f"\nText: \"{text}\"")
        print(f"  Prediction: {labels.get(prediction, 'Unknown')}")
        print(f"  Confidence: {max(probability):.2%}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved as: model.pkl")
    print(f"Vectorizer saved as: vectorizer.pkl")
    print(f"\nYou can now run 'python app.py' to start the web application.")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()

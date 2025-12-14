# 🛡️ Cyberbullying Detection Using Machine Learning

A web-based application that detects cyberbullying content in text using Natural Language Processing (NLP) and Machine Learning (ML) techniques.

## 📋 Project Overview

This project is developed as a **Major Project for 5th Semester** and demonstrates the application of ML/NLP in creating safer online environments.

### Key Features

- ✅ Real-time text analysis for cyberbullying detection
- ✅ NLP preprocessing pipeline (tokenization, lemmatization, stopword removal)
- ✅ TF-IDF vectorization for feature extraction
- ✅ Multiple ML classifiers comparison
- ✅ User-friendly web interface
- ✅ Confidence score display
- ✅ Comprehensive project documentation pages

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | HTML5, CSS3, JavaScript |
| Backend | Python Flask |
| ML Library | Scikit-learn |
| NLP Library | NLTK |
| Feature Extraction | TF-IDF |

## 📁 Project Structure

```
cyberbullying-detection/
│
├── app.py                 # Flask application (main backend)
├── train_model.py         # Model training script
├── model.pkl              # Trained ML model (generated)
├── vectorizer.pkl         # TF-IDF vectorizer (generated)
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── templates/             # HTML templates
│   ├── index.html         # Home page
│   ├── introduction.html  # Introduction page
│   ├── objectives.html    # Project objectives
│   ├── feasibility.html   # Feasibility study
│   ├── srs.html           # Software Requirements
│   ├── methodology.html   # Methodology page
│   ├── future_scope.html  # Future enhancements
│   └── detect.html        # Detection tool page
│
└── static/                # Static files
    ├── style.css          # Main stylesheet
    └── script.js          # JavaScript functionality
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Navigate to Project Directory

```bash
cd "c:\Users\rohit\Desktop\5th Sem\Project for IT\Simran\cyberbullying-detection"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
python train_model.py
```

This will:
- Create a sample dataset (or use existing dataset if available)
- Preprocess the text data
- Train multiple ML models
- Select the best performing model
- Save `model.pkl` and `vectorizer.pkl`

### Step 5: Run the Application

```bash
python app.py
```

### Step 6: Open in Browser

Navigate to: **http://127.0.0.1:5000**

## 📖 Website Pages

| Page | Description |
|------|-------------|
| Home | Project introduction and overview |
| Introduction | Cyberbullying explanation and impact |
| Objectives | Project goals and expected outcomes |
| Feasibility | Technical, operational, economic, social feasibility |
| SRS | Software Requirement Specification |
| Methodology | Step-by-step development process |
| Future Scope | Potential enhancements |
| Detect | Live detection tool |

## 🧪 Using the Detection Tool

1. Navigate to the **Detect** page
2. Enter text in the textarea
3. Click **Analyze Text**
4. View the classification result:
   - ✅ **Non-Bullying**: Safe content
   - ❌ **Cyberbullying Detected**: Harmful content
   - ⚠️ **Potentially Harmful**: Mildly negative content

## 📊 Model Performance

The system compares multiple ML algorithms:

| Model | Typical Accuracy |
|-------|------------------|
| Logistic Regression | ~85-90% |
| SVM | ~86-91% |
| Naive Bayes | ~80-85% |
| Random Forest | ~84-88% |

*Note: Actual performance depends on the dataset used.*

## 🔧 Using Your Own Dataset

Place a CSV file named `dataset.csv` in the project directory with columns:
- `text`: The text content
- `label`: 0 (non-bullying), 1 (bullying), 2 (neutral)

Then run `python train_model.py` to retrain the model.

## 📝 API Endpoint

**POST /predict**

```json
Request:
{
    "text": "Your text here"
}

Response:
{
    "prediction": "Cyberbullying Detected",
    "label": "bullying",
    "confidence": 0.92,
    "message": "This text contains potentially harmful content.",
    "original_text": "Your text here"
}
```

## 👥 Team

- **Project Title**: Cyberbullying Detection Using Machine Learning
- **Semester**: 5th Semester
- **Subject**: Major Project / IT Project

## 📚 References

1. Twitter Cyberbullying Dataset - Kaggle
2. NLTK Documentation - https://www.nltk.org/
3. Scikit-learn Documentation - https://scikit-learn.org/
4. Flask Documentation - https://flask.palletsprojects.com/

## 📄 License

This project is developed for educational purposes as part of college curriculum.

---

**© 2025 Cyberbullying Detection Project. All Rights Reserved.**

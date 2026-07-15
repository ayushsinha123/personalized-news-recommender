# 📰 Personalized News Recommendation System

An AI-powered **Personalized News Recommendation System** that combines **semantic retrieval** using Sentence Transformers and **FAISS** with **Machine Learning ranking** using Logistic Regression and LightGBM to deliver personalized news recommendations based on user reading history.

---

# 🚀 Live Demo

### 🌐 Streamlit Application

https://personalized-news-recommender-by-ayush-sinha.streamlit.app/

---

# 📂 GitHub Repository

https://github.com/ayushsinha123/personalized-news-recommender

---

# 📖 Project Overview

Modern recommendation systems are built in **multiple stages** instead of directly recommending items.

This project follows a **two-stage recommendation architecture**:

1. Generate semantic embeddings for news articles.
2. Build personalized user profiles from reading history.
3. Retrieve candidate articles using FAISS vector similarity search.
4. Engineer ranking features.
5. Re-rank retrieved candidates using Machine Learning.
6. Display the final personalized recommendations through an interactive Streamlit application.

The project is inspired by recommendation pipelines used in modern content platforms.

---

# ✨ Features

- 📰 Personalized News Recommendations
- 🤖 AI-based Semantic Retrieval
- 🔍 FAISS Vector Search
- 🧠 Sentence Transformer Embeddings
- 📊 Feature Engineering Pipeline
- 📈 Logistic Regression Ranking
- ⚡ LightGBM Ranking
- 👤 User Profile Generation
- ❄️ Cold Start Recommendation Support
- 💡 Explainable Recommendations
- 🌐 Interactive Streamlit Web Application
- 📱 Modern Responsive Interface

---

# 🏗️ Recommendation Pipeline

```text
                 User Reading History
                         │
                         ▼
               User Profile Embedding
                         │
                         ▼
      Sentence Transformer Embeddings
                         │
                         ▼
          FAISS Candidate Retrieval
                         │
                         ▼
             Feature Engineering
                         │
                         ▼
      Logistic Regression / LightGBM
                         │
                         ▼
        Final Personalized Ranking
                         │
                         ▼
              Recommended Articles
```

---

# 🧠 Technologies Used

| Category | Technologies |
|-----------|--------------|
| Programming | Python |
| Machine Learning | Scikit-learn, LightGBM |
| NLP | Sentence Transformers |
| Vector Search | FAISS |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Dataset | Microsoft MIND Dataset |

---

# 📂 Project Structure

```text
personalized-news-recommender/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── processed_news.csv
│   ├── processed_behaviors.csv
│   ├── news_embeddings.npy
│   ├── user_embeddings.npy
│   ├── news_id_to_idx.pkl
│   ├── faiss.index
│   ├── popularity_scores.csv
│   ├── ranker_logreg.pkl
│   └── ranker_lightgbm.pkl
│
├── notebooks/
│   ├── 01_Exploration.ipynb
│   ├── 02_Model_Development.ipynb
│   └── 03_Evaluation.ipynb
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── embedding.py
│   ├── user_profile.py
│   ├── ranking.py
│   ├── ranker.py
│   ├── recommender.py
│   ├── evaluation.py
│   └── explainability.py
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

This project uses the **Microsoft MIND (Microsoft News Dataset)**.

The dataset contains:

- News Articles
- User Reading History
- Click Behaviors
- Impression Logs
- News Categories
- Subcategories

---

# ⚙️ Machine Learning Pipeline

## Candidate Retrieval

- Sentence Transformer (all-MiniLM-L6-v2)
- FAISS IndexFlatIP

---

## Ranking Models

### Logistic Regression

Uses engineered ranking features to estimate click probability.

### LightGBM

Gradient Boosted Decision Trees trained for personalized ranking.

---

# 📈 Engineered Features

The ranking model uses:

- Similarity Score
- Popularity Score
- Category Match
- Subcategory Match
- Category Affinity
- User History Length
- Maximum Similarity to History
- Title Length
- Abstract Length

---

# 🖥️ Application Features

The deployed Streamlit application includes:

- Existing User Recommendation
- Cold Start Recommendation
- Article Search
- Recommendation Explainability
- Interactive Dashboard
- Dataset Statistics
- Recommendation Pipeline Visualization

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/ayushsinha123/personalized-news-recommender.git
```

Move into the project

```bash
cd personalized-news-recommender
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/streamlit_app.py
```

---

# 📸 Application Screenshots

## Home Page

> *(Add screenshot here)*

```text
assets/home.png
```

---

## Recommendations

> *(Add screenshot here)*

```text
assets/recommendations.png
```

---

## Architecture

> *(Add screenshot here)*

```text
assets/architecture.png
```

---

## About

> *(Add screenshot here)*

```text
assets/about.png
```

---

# 🔮 Future Improvements

- Neural Learning-to-Rank Models
- Real-time User Feedback Learning
- Online User Profile Updates
- Multi-language News Recommendation
- Hybrid Collaborative + Content-Based Recommendation
- Incremental Model Retraining
- Explainable AI using SHAP

---

# 👨‍💻 Author

## Ayush Sinha

**B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)**  
Siksha 'O' Anusandhan University

### 🔗 GitHub

https://github.com/ayushsinha123

### 💼 LinkedIn

https://www.linkedin.com/in/ayush-sinha-aiml/

### 🌐 Live Demo

https://personalized-news-recommender-by-ayush-sinha.streamlit.app/

---

# ⭐ If you found this project interesting, consider giving it a star!
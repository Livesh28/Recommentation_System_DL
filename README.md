# 🌱 AI Recommendation Engine

### Intelligent Eco-Friendly Product Recommendation System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-FF6F00?style=for-the-badge&logo=tensorflow)
![FAISS](https://img.shields.io/badge/FAISS-Semantic_Search-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

# 📖 Overview

The **AI Recommendation Engine** is the core intelligence module of the **AI-Powered Sustainable Product Recommendation System**.

Traditional recommendation systems mainly focus on popularity and sales. This project introduces an **AI-powered Hybrid Recommendation System** that recommends products based on:

- 🌱 Sustainability
- 🌍 Carbon Footprint
- 💚 Eco Score
- 🛍 Product Similarity
- 🤖 AI Semantic Search

The objective is to encourage environmentally conscious purchasing decisions.

---

# ✨ Features

## 🤖 AI Recommendation Engine

- Semantic Product Search
- Similar Product Recommendation
- Personalized Recommendations
- Hybrid Recommendation System

---

## 🌍 Sustainability Analysis

- Carbon Footprint Prediction
- Sustainability Score
- Eco-Friendly Rating
- Green Product Suggestions

---

## 🔍 Intelligent Search

- Search by Product Name
- Search by Category
- Search Similar Products
- AI Text Search

---

## 📊 Dashboard

- Product Analytics
- Sustainability Metrics
- Carbon Emission Display
- Recommendation Results

---

# 🧠 AI Technologies Used

| Technology | Purpose |
|------------|---------|
| Sentence Transformers | Semantic Embeddings |
| FAISS | Similarity Search |
| Scikit-Learn | Machine Learning |
| TensorFlow | Deep Learning |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| FastAPI | REST API |
| Joblib | Model Storage |

---

# 🏗 Project Architecture

```
                   User
                     │
                     ▼
              Product Search
                     │
                     ▼
        Sentence Transformer Model
                     │
                     ▼
             Generate Embeddings
                     │
                     ▼
             FAISS Similarity Search
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 Carbon Prediction      Recommendation Engine
          │                     │
          └──────────┬──────────┘
                     ▼
          Sustainable Product List
```

---

# 📂 Project Structure

```
recommendation-system/

│
├── static/
│
├── app.py
├── recommender.py
├── bootstrap.py
│
├── processed_products.csv
├── product_embeddings.npy
├── interactions.csv
│
├── carbon_credit_model.joblib
├── eco_model.joblib
├── preprocessor.joblib
├── feature_columns.joblib
│
├── faiss_product_index.bin
│
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/recommendation-system.git
```

## Enter Project

```bash
cd recommendation-system
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Start FastAPI Server

```bash
uvicorn app:app --reload
```

Application

```
http://127.0.0.1:8000
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

# 📊 Machine Learning Models

| Model | Description |
|--------|-------------|
| carbon_credit_model.joblib | Carbon Footprint Prediction |
| eco_model.joblib | Sustainability Prediction |
| preprocessor.joblib | Data Preprocessing |
| feature_columns.joblib | Feature Encoding |
| faiss_product_index.bin | Semantic Search Index |

---

# 📈 Recommendation Workflow

```
User Query
     │
     ▼
Product Embedding
     │
     ▼
Sentence Transformer
     │
     ▼
FAISS Similarity Search
     │
     ▼
Top Similar Products
     │
     ▼
Carbon Prediction
     │
     ▼
Sustainability Score
     │
     ▼
Recommended Products
```

---

# 🛠 Tech Stack

### Backend

- FastAPI
- Python

### Machine Learning

- Scikit-Learn
- TensorFlow

### AI

- Sentence Transformers
- FAISS

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Model Storage

- Joblib

---

# 📦 Dataset

The project is trained using:

- Sustainable Product Dataset
- Product Metadata
- Carbon Footprint Dataset
- User Interaction Dataset

---

# 🚀 Future Enhancements

- ✅ Explainable AI (XAI)
- ✅ Deep Learning Ranking Model
- ✅ User Authentication
- ✅ Personalized Dashboard
- ✅ Cloud Deployment
- ✅ Mobile App
- ✅ Real-Time Recommendation
- ✅ Product Image Recognition
- ✅ LLM-powered Product Assistant

---

# 📸 Screenshots

> Add your project screenshots here.

```
images/dashboard.png

images/recommendation.png

images/product-search.png
```

---

# 👨‍💻 Author

**Livesh L**

Artificial Intelligence & Data Science

Prathyusha Engineering College

---

# 📜 License

Licensed under the **MIT License**.

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It helps others discover the project!

# 🛒 Shopper Spectrum
### Customer Segmentation & Product Recommendation in E-Commerce

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shopperspectrum-9i7au2q9t3whdzje2dvch6.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![GitHub](https://img.shields.io/badge/GitHub-abhi--1009-black)

---

## 📌 Project Overview

**Shopper Spectrum** is an end-to-end machine learning project that analyzes real-world e-commerce transaction data to:

- 🔍 **Segment customers** into meaningful groups using RFM Analysis + KMeans Clustering
- 🎯 **Recommend products** using Item-Based Collaborative Filtering
- 📊 **Visualize insights** through an interactive Streamlit web application

> **Live App:** https://shopperspectrum-9i7au2q9t3whdzje2dvch6.streamlit.app/
---

## 🧠 Problem Statement

The global e-commerce industry generates vast amounts of transaction data daily. Without proper analysis, businesses:
- Treat all customers the same (wasting marketing budget)
- Miss cross-selling opportunities
- Fail to identify at-risk customers before they churn

This project solves these problems using unsupervised machine learning and collaborative filtering.

---

## 📂 Project Structure

```
Shopper_Spectrum/
│
├── shopper_spectrum_streamlit.py   # Streamlit web application (4 pages)
├── shopper_spectrum.py             # Training pipeline (EDA + ML + saving models)
│
├── kmeans_model.pkl                # Trained KMeans model (k=4)
├── scaler.pkl                      # Fitted StandardScaler for RFM normalization
├── rfm_data.csv                    # RFM table with cluster labels per customer
├── online_retail.csv               # Raw dataset (stored via Git LFS)
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes large pkl and EDA chart files
└── .gitattributes                  # Git LFS tracking config
```

> **Note:** `item_similarity.pkl` is excluded from GitHub due to its large size. It is automatically rebuilt on the first app load using `online_retail.csv`.

---

## 📊 Dataset

**Source:** UCI Online Retail Dataset

| Column | Description |
|---|---|
| InvoiceNo | Transaction number (prefix 'C' = cancellation) |
| StockCode | Unique product code |
| Description | Product name |
| Quantity | Units purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit (£) |
| CustomerID | Unique customer identifier |
| Country | Customer's country |

**Raw Dataset Stats:**
- Total Records: 541,909
- Missing CustomerIDs: 135,080 (24.9%)
- Duplicate Records: 5,268
- **After Cleaning: 392,692 records**

---

## 🔧 Project Workflow

### Step 1 — Data Preprocessing
- Parse `InvoiceDate` to datetime
- Drop rows with missing `CustomerID`
- Remove cancelled invoices (InvoiceNo starting with `C`)
- Remove invalid quantities and prices (≤ 0)
- Drop duplicate records
- Add `TotalPrice = Quantity × UnitPrice`

### Step 2 — Exploratory Data Analysis (EDA)
4 charts saved to `EDA_Charts/`:
1. Top 10 Countries by Transaction Volume
2. Top 20 Selling Products
3. Monthly Revenue Trend
4. Customer Spend Distribution

### Step 3 — RFM Feature Engineering

| Feature | Definition | Calculation |
|---|---|---|
| **Recency** | Days since last purchase | Snapshot Date − Last Purchase Date |
| **Frequency** | Number of unique invoices | Count of unique InvoiceNo per customer |
| **Monetary** | Total amount spent | Sum of TotalPrice per customer |

### Step 4 — KMeans Clustering

**Model Selection:**
- Elbow Method → clear bend at k=4
- Silhouette Score → 0.6162 at k=4 (highest)

**Customer Segments:**

| Cluster | Segment | Recency (avg) | Frequency (avg) | Monetary (avg) |
|---|---|---|---|---|
| 2 | 🟢 High-Value | 7.4 days | 82.5 purchases | £127,188 |
| 0 | 🟡 Regular | 43.7 days | 3.7 purchases | £1,354 |
| 3 | 🔵 Occasional | 15.5 days | 22.3 purchases | £12,691 |
| 1 | 🔴 At-Risk | 248.1 days | 1.6 purchases | £479 |

### Step 5 — Product Recommendation System
- Built a **CustomerID × Product** pivot matrix
- Applied **Cosine Similarity** between product vectors
- Returns **Top 5 similar products** for any given product name

**Example:**
```
Input:  GREEN VINTAGE SPOT BEAKER
Output: BLUE VINTAGE SPOT BEAKER
        PINK VINTAGE SPOT BEAKER
        POTTING SHED CANDLE CITRONELLA
        POTTING SHED ROSE CANDLE
        PANTRY CHOPPING BOARD
```

---

## 📱 Streamlit App Pages

| Page | Features |
|---|---|
| 🏠 Home | Project overview, segment definitions table |
| 📊 RFM Analysis | 4 distribution charts + segment summary table |
| 👤 Customer Segmentation | Input R, F, M values → Predict segment with business tip |
| 🎯 Product Recommendation | Input product name → Get 5 similar products |

---

## 🚀 How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/abhi-1009/shopper_spectrum.git
cd shopper_spectrum
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the training pipeline first** (generates model files):
```bash
python shopper_spectrum.py
```

**4. Launch the Streamlit app:**
```bash
streamlit run shopper_spectrum_streamlit.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (KMeans, StandardScaler, Cosine Similarity) |
| Web App | Streamlit |
| Version Control | Git + GitHub + Git LFS |
| Deployment | Streamlit Community Cloud |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Optimal Clusters (k) | 4 |
| Best Silhouette Score | 0.6162 |
| Customers Segmented | 4,300+ unique customers |
| Products in Similarity Matrix | 3,000+ unique products |

---

## 🌐 Deployment

The app is deployed on **Streamlit Community Cloud**:

- **Repository:** `abhi-1009/shopper_spectrum`
- **Branch:** `main`
- **Main file:** `shopper_spectrum_streamlit.py`
- **Live URL:** https://shopperspectrum-9i7au2q9t3whdzje2dvch6.streamlit.app/

> `online_retail.csv` is stored using **Git LFS** due to its 49MB size.
> `item_similarity.pkl` is auto-rebuilt on first app load — no manual upload needed.

---

## 📌 Key Business Insights

- 🇬🇧 **UK dominates 85%+** of transactions — international expansion is a major opportunity
- 📅 **November revenue spikes** sharply — stock up by October, run promotions in September
- 💎 **High-Value customers** are rare but generate massive revenue — protect and reward them
- ⚠️ **At-Risk customers** haven't purchased in ~248 days on average — urgent win-back campaigns needed
- 🛍️ **Top products** are home decor and gift items — strong cross-selling potential

---

## 👨‍💻 Author

**Abhijit Sinha** — [github.com/abhi-1009](https://github.com/abhi-1009)

---

*Shopper Spectrum — End-to-End E-Commerce Analytics | June 2026*


<img width="373" height="284" alt="KNN" src="https://github.com/user-attachments/assets/5cb9ccfc-ec67-4858-a5da-e45f75d1efb8" />
<img width="373" height="284" alt="KNN" src="https://github.com/user-attachments/assets/e55e6564-7904-49a4-bb2c-fad86d324fd7" />
<img width="373" height="284" alt="KNN" src="https://github.com/user-attachments/assets/7f49e171-5bb9-4c03-947e-dd1720719f72" />
<div align="center">

# 📨 Message Intelligence System

### *Probability-driven spam detection powered by machine learning*

<br/>

[![Streamlit App](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://message-intelligence-system-supervised-learning.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Engine-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![Dataset](https://img.shields.io/badge/📊%20Dataset-5200%20Messages-06B6D4?style=for-the-badge)](https://message-intelligence-system-supervised-learning.streamlit.app/)

<br/>

> An interactive machine learning dashboard that classifies messages as **spam or legitimate** using three tuned classifiers —
> K-Nearest Neighbors, Support Vector Machine, and Naive Bayes — trained on **5,200 real-world-style messages**
> with engineered behavioral and textual features.

<br/>

</div>

---

## 🌐 Live Application

<div align="center">

### 👉 [https://message-intelligence-system-supervised-learning.streamlit.app/](https://message-intelligence-system-supervised-learning.streamlit.app/)

*No installation needed — open the link, type a message, and get an instant spam verdict with confidence score.*

</div>

---

## ✨ App Tabs at a Glance

| Tab | What you get |
|-----|-------------|
| 🏠 **Overview** | Dataset snapshot, class balance donut chart, sample message table |
| 📊 **Explore Data** | Feature distributions, box plots, correlation heatmap, spam timing charts |
| 🤖 **Model Training** | KNN K-sweep, SVM kernel search, Naive Bayes posterior probabilities, confusion matrices |
| 📈 **Comparison** | Side-by-side Accuracy / Precision / Recall / F1 for all three models |
| 🔮 **Live Predictor** | Real-time classification with probability gauge and confidence score |

---

## 🧠 Models & Results

Three supervised learning classifiers were trained and hyper-parameter tuned with `GridSearchCV`:

```
KNN  ──── Best K = 5              ──── Accuracy: 100.00%
SVM  ──── Best kernel (RBF)       ──── Accuracy: 100.00%
NB   ──── var_smoothing optimized ──── Accuracy: 100.00%
```

---

### 📊 KNN — K Value Sweep (K = 1 to 20)

Accuracy stays perfect up to **K = 5**, then stabilises at ~99.90% — making K = 5 the optimal sweet spot before bias increases.

<div align="center">
<img width="373" height="284" alt="KNN" src="https://github.com/user-attachments/assets/e3ff345e-686d-4ef9-8c13-87eebd8c6a6e" />
</div>

---

### ⚔️ SVM — Margin Separation & Support Vectors

The SVM with a linear kernel projects the data into 2D PCA space, creating a clean decision boundary between spam (red) and legitimate (blue) clusters. Circled points are the **support vectors** defining the maximum margin.

<div align="center">
<img width="430" height="341" alt="SVM" src="https://github.com/user-attachments/assets/33df9e91-0b40-4773-b26d-262069bc8603" />
</div>


---

### 🏆 Model Comparison — KNN vs SVM vs Naive Bayes

All three classifiers achieve **100% accuracy** on the test set, validating the strength of the engineered feature set.

<div align="center">
<img width="385" height="296" alt="Knn_Svm_Naivebayes" src="https://github.com/user-attachments/assets/9ab0d063-13c2-4d5e-9c58-8853cfd103e5" />
</div>

---

## 🗂️ Dataset

| Property | Value |
|----------|-------|
| 📄 **File** | `Message_Intelligence_Dataset_5200_.csv` |
| 🔢 **Samples** | 5,200 labeled messages |
| ⚖️ **Balance** | ~50% Spam / ~50% Legitimate |
| 🔬 **Features** | 12 engineered columns + 1 target |
| 🔗 **Access** | [View via Live App →](https://message-intelligence-system-supervised-learning.streamlit.app/) |

### 📐 Feature Columns

The dataset combines **textual signals** with **sender behavioral metadata**:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `message_length` | Textual | Total character count |
| 2 | `word_count` | Textual | Number of words |
| 3 | `num_urls` | Textual | Hyperlinks detected via regex |
| 4 | `num_digits` | Textual | Count of numeric characters |
| 5 | `num_special_chars` | Textual | Non-alphanumeric, non-space characters |
| 6 | `spam_keyword_score` | Textual | Hit count against 30+ spam keywords |
| 7 | `legit_keyword_score` | Textual | Hit count against legitimate vocabulary |
| 8 | `sender_activity_score` | Behavioral | Activity signal of sender account |
| 9 | `sender_account_age_days` | Behavioral | Days since account was created |
| 10 | `messages_sent_last_24h` | Behavioral | Sending velocity in last 24 hours |
| 11 | `hour_of_day` | Temporal | Hour the message was sent (0–23) |
| 12 | `day_of_week` | Temporal | Day of the week (0 = Mon, 6 = Sun) |

**Target:** `spam_label` → `0` = Legitimate &nbsp;|&nbsp; `1` = Spam

---

## 🏗️ Project Structure

```
📦 message-intelligence-system/
│
├── 📄 app.py                                  # Streamlit dashboard (main entry point)
├── 📓 Message_Intelligence_System.ipynb       # Original analysis notebook
├── 📊 Message_Intelligence_Dataset_5200_.csv  # Training dataset (5,200 messages)
│
├── 🖼️  KNN.png                                # KNN K-value accuracy curve
├── 🖼️  SVM.png                                # SVM margin separation plot
├── 🖼️  Knn_Svm_Naivebayes.png                # Model comparison bar chart
│
└── 📋 requirements.txt                        # Python dependencies
```

---

## 🚀 Run Locally

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/message-intelligence-system.git
cd message-intelligence-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

App opens at **`http://localhost:8501`**

### `requirements.txt`

```txt
streamlit>=1.50
pandas
numpy
scikit-learn
plotly
```

---

## 🔮 Live Predictor — How It Works

1. **Type a message** → e.g. *"Claim your free prize now before it expires!"*
2. **Tune sender signals** → account age, activity score, sending velocity, hour of day
3. **Pick a classifier** → SVM · Naive Bayes · KNN
4. **Hit Classify** → instant verdict + probability gauge

Text features (`spam_keyword_score`, `num_urls`, `message_length` …) are **auto-derived** from your input. The same `StandardScaler` used during training is applied before inference.

**Built-in quick-test examples:**

| Button | Message |
|--------|---------|
| 💸 Prize scam | *"Congratulations! You've been selected for a cash prize…"* |
| 📎 Work email | *"Hi, can we reschedule our meeting to 10:30 AM tomorrow?"* |
| 🏦 Loan spam | *"Get instant cashback and a free gift card, act fast…"* |

---

## ⚙️ ML Pipeline

```
Raw CSV ──► StandardScaler ──► Train / Test Split (80 / 20, stratified)
                │
                ├──► KNN  (K sweep 1–20, best K auto-selected)
                ├──► SVM  (GridSearchCV: kernel × C × gamma, cv=5)
                └──► NB   (GridSearchCV: var_smoothing, cv=5)
                                    │
                                    ▼
                      Accuracy · Precision · Recall · F1
                      Confusion Matrices · PCA Visualization
```

| Detail | Value |
|--------|-------|
| Train / Test split | 80% / 20%, stratified |
| Preprocessing | `StandardScaler` (z-score normalization) |
| Hyperparameter tuning | 5-fold `GridSearchCV` |
| SVM visualization | 2-component PCA projection |
| Caching | `@st.cache_data` + `@st.cache_resource` |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Interactive web dashboard |
| [scikit-learn](https://scikit-learn.org) | KNN · SVM · Naive Bayes · PCA · GridSearchCV |
| [Plotly](https://plotly.com) | Interactive dark-themed charts |
| [Pandas](https://pandas.pydata.org) | Data loading & manipulation |
| [NumPy](https://numpy.org) | Numerical operations |

---

## 📈 Key Insights

- 🔑 **`spam_keyword_score`** and **`message_length`** are the strongest discriminating features
- 🕐 **Spam peaks between 9 AM – 12 PM** and again in late evening
- 👤 **New accounts (< 90 days old)** + **high send velocity (> 20 msgs / 24h)** are the strongest behavioral spam signals
- ✅ All three classifiers perform near-identically, confirming the feature engineering captures the signal perfectly

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute with attribution.

---

<div align="center">

**Built with** 💜 **using Streamlit · scikit-learn · Plotly**

*Message Intelligence System — KNN · SVM · Naive Bayes*

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Open%20Now-FF4B4B?style=flat-square&logo=streamlit)](https://message-intelligence-system-supervised-learning.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikitlearn)](https://scikit-learn.org)

</div>

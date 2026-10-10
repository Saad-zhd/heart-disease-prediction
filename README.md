# ❤️ Heart Disease Prediction System
### An End-to-End Machine Learning Diagnostic System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-prediction-6164.streamlit.app/)

> **Live Application URL:** [https://heart-disease-prediction-6164.streamlit.app/](https://heart-disease-prediction-6164.streamlit.app/)

---

## 📌 Project Overview
This project builds a complete end-to-end Machine Learning system for predicting the risk of coronary heart disease using the **UCI Heart Disease benchmark dataset**. It implements the full lifecycle of a machine learning application:
1. **Data Understanding & Preprocessing** (Missing value inspection, clinical attribute formatting)
2. **Feature Extraction & Discrimination Analysis** (Selecting the 4 most discriminating clinical attributes)
3. **Deterministic Label Encoding** (Converting categorical text to numeric vectors)
4. **Model Training** (Support Vector Classifier, SVC with radial basis kernel)
5. **Model Evaluation** (Accuracy, Precision, Recall, F1-Score, and Confusion Matrix heatmap)
6. **Application Phase** (Interactive and programmatic inference via PrettyTable)
7. **Feedback Phase** (Cardiology clinical review and continuous improvement roadmap)
8. **Cloud Deployment** (Interactive web application on Streamlit Community Cloud)

---

## 📊 4 Selected Most Discriminating Clinical Attributes

Through statistical correlation analysis ($|r|$), mutual information, and clinical significance, the 4 top discriminating features were selected:

| Attribute | Clinical Description | Categories / Range | Discrimination Factor |
| :--- | :--- | :--- | :--- |
| **`ChestPain`** | Type of chest discomfort | Typical Angina, Atypical Angina, Non-anginal Pain, Asymptomatic | $|r| = 0.435$ (Primary symptom of coronary insufficiency) |
| **`Gender`** | Biological gender | Female, Male | $|r| = 0.280$ (Well-established demographic disparity in cardiovascular risk) |
| **`ExerciseAngina`** | Exercise-induced angina | No, Yes | $|r| = 0.438$ (Direct physiological indicator of myocardial ischemia) |
| **`MajorVessels`** | Major coronary vessels colored by fluoroscopy | Zero, One, Two, Three, Four | $|r| = 0.382$ (Anatomical measure of arterial narrowing) |

- **Target Output (`HeartDisease`)**: `No` (Absence / Healthy: 0) or `Yes` (Presence / Disease: 1)
- **Held-out Test Performance**: **81% Accuracy**, with balanced sensitivity (0.83 recall on positive disease cases) and specificity (0.79 recall on healthy cases).

---

## 📁 Repository Structure

```text
heart-disease-prediction/
├── 1 - Heart Disease Prediction/            # Notebook, Datasets, and Model Artifacts
│   ├── Heart_Disease_Prediction.ipynb       # Complete executed Jupyter Notebook
│   ├── heart_disease_model.pkl              # Serialized trained SVC model (.pkl)
│   ├── svc_trained_model.pkl                # Companion serialized model (.pkl)
│   ├── processed_cleveland.csv              # Benchmark UCI Cleveland dataset (.csv)
│   ├── sample-data.csv                      # Formatted 4-attribute dataset (.csv)
│   ├── sample-data-encoded-output.csv       # Intermediate dataset with encoded target (.csv)
│   ├── sample-data-encoded.csv              # Fully numeric encoded dataset (.csv)
│   ├── training-data-encoded.csv            # 80% Training set (.csv)
│   ├── testing-data-encoded.csv             # 20% Testing set (.csv)
│   └── model-predictions.csv                # Model testing predictions and true labels (.csv)
├── streamlit-app/                           # Cloud Web Application
│   ├── app.py                               # Streamlit web application frontend
│   ├── svc_trained_model.pkl                # Deployed model for web app
│   ├── requirements.txt                     # Streamlit cloud dependencies
│   └── runtime.txt                          # Python runtime definition (3.11)
├── .gitignore                               # Excludes PDFs, ZIPs, and environments
└── README.md                                # Project overview and documentation
```

---

## 🚀 How to Run the Project Locally

### 1. Setup Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell
# OR venv\Scripts\activate.bat on CMD
```

### 2. Install Dependencies
```powershell
pip install numpy pandas scikit-learn matplotlib seaborn prettytable streamlit
```

### 3. Open Jupyter Notebook
```powershell
cd "1 - Heart Disease Prediction"
jupyter notebook
```
Open `Heart_Disease_Prediction.ipynb` and run all cells (**Kernel -> Restart & Run All**).

### 4. Run Streamlit Web Application Locally
```powershell
cd streamlit-app
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to interact with the web prediction system.

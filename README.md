## 📊 Student Performance Prediction

## 📌 Project Overview

  This project predicts student performance scores using machine learning regression techniques.
The system analyzes key factors such as:

   • Gender                                
   • Race/Ethnicity                                                       
   • Parental Level of Education                                                      
   • Lunch Type                                                            
   • Test Preparation Course                                                         
   • Reading Score                                                                          
   • Writing Score                                                                    

The goal is to build a robust regression pipeline that can generalize well and provide accurate predictions of a student's math score (or overall academic performance).

The project includes:
✅ **Multiple regression algorithms** (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, etc.)
✅ **ML Pipelines** for preprocessing + modeling
✅ **Flask web application** for user interaction
✅ **AWS-ready configuration** for cloud deployment
✅ **Dockerized Deployment** for Run anywhere with a single container.

## 📈 Example:

    gender	  race/ethnicity	parental level	    lunch	     prep course	   reading score    	writing score
   ____________________________________________________________________________________________________________
    female	     group B	      bachelor's	     standard	    completed	          72	            74

✅ Output: Predicted Math Score = 78.5

## 🚀 Features

• End-to-end ML pipeline with data ingestion, transformation, model training, and evaluation.
• Comparative analysis of different regression models.
• Hyperparameter tuning for performance optimization.
• Flask web interface to input student details and get predicted scores in real time.
• Modular project structure with clear separation of concerns.
• Dockerized Deployment for Run anywhere with a single container.
• Scalable design for cloud deployment (AWS EC2 / Elastic Beanstalk / S3 integration).

## 🛠️ Tech Stack

• Python 3.11+
• Flask – Web framework
• Scikit-learn – Preprocessing, training & evaluation
• XGBoost / CatBoost – Advanced regression models
• Dill – Model serialization
• Docker – Containerization
• AWS CLI – Cloud deployment ready

## 📂 Project Structure

    Student-Performance/
    │── .ebextensions
    │   └── python.config
    │── .gitignore
    │── application.py              # Flask app entry point
    │── Dockerfile                  # Docker container setup
    │── requirements.txt            # Python dependencies
    │── setup.py                    # Package setup
    │── notebook/                   # Jupyter notebooks & raw dataset
    │   └── data/StudentsPerformance.csv
    │   └── 1. EDA STUDENT PERFORMANCE.ipynb
    │   └── 2. MODEL TRAINING.ipynb
    │── templates/                  # HTML templates for Flask
    │   ├── index.html
    │   └── home.html
    │── Source/                     # Core ML pipeline package
    │   ├── Components/
    │   │   ├── data_ingestion.py   # Data ingestion & train/test split
    │   │   ├── data_transformation.py # Preprocessing pipelines
    │   │   └── model_trainer.py    # Model training & evaluation
    │   ├── pipeline/
    │   │   └── predict_pipeline.py # Inference pipeline for predictions
    │   │   └── train_pipeline.py   # Inference pipeline for training
    │   ├── exception.py            # Custom exception handling
    │   ├── logger.py               # Logging utility
    │   └── utils.py                # Helper functions (save/load/eval)
    │── artifacts/                  # Saved artifacts (generated)
    │   ├── model.pkl               # Trained ML model
    │   └── preprocessor.pkl        # Preprocessing pipeline
    │── logs/                       # Log files


## ⚙️ Installation & Setup

## 1️⃣ Clone Repository

    git clone https://github.com/THOWFI/Student-Performance.git
    cd Student-Performance

## 2️⃣ Create Virtual Environment

    python -m venv venv
    source venv/bin/activate   # On Linux/Mac
    venv\Scripts\activate      # On Windows

## 3️⃣ Install Dependencies

    pip install -r requirements.txt

## 4️⃣ Run Flask App

    python application.py


To Check Visit:
👉 http://127.0.0.1:5000/

To Check Predict:
👉 http://127.0.0.1:5000/predictdata

## 🧩 Pipeline Workflow

1. **Data Ingestion** (`data_ingestion.py`)

   • Reads raw dataset (`StudentsPerformance.csv`).
   • Splits into train/test datasets.
   • Stores CSVs in `artifacts/`.

2. **Data Transformation** (`data_transformation.py`)

   • Handles missing values with `SimpleImputer`.
   • Scales numerical columns (`StandardScaler`).
   • Encodes categorical features (`OneHotEncoder`).
   • Saves preprocessing pipeline as `preprocessor.pkl`.

3. **Model Training** (`model_trainer.py`)

   • Trains multiple regression algorithms.
   • Uses **GridSearchCV/RandomizedSearchCV** for hyperparameter tuning.
   • Selects best model and saves as `model.pkl`.

4. **Prediction Pipeline** (`predict_pipeline.py`)

   • Loads trained model + preprocessor.
   • Accepts user input, preprocesses, and predicts math score.

5. **Flask Application** (`application.py`)

   • Web form to input features.
   • Displays prediction results.

6. **Custom Utilities**

   • `exception.py`: Custom structured exception handling.
   • `logger.py`: Timestamped logging with rotating logs.
   • `utils.py`: Save/load objects, evaluate models.

## 🔮 Future Enhancements

• 📊 **Model Explainability** – SHAP/LIME for feature importance.
• ☁️ **Full AWS Deployment** – Elastic Beanstalk / ECS integration.
• 📱 **Frontend Upgrade** – Interactive dashboard (React/Streamlit).
• 🎯 **Multi-Target Prediction** – Extend to predict reading & writing scores.

## 🐳 Docker Hub Link

https://hub.docker.com/r/thowfiq03/studentperformance


## 📜 License

This project is not licensed for public use.

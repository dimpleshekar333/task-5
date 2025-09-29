
# Heart Disease Prediction using Decision Tree & Random Forest

## 📌 Overview
This project demonstrates how to use **Decision Tree** and **Random Forest** classifiers to predict heart disease based on patient health attributes.  
We also visualize the Decision Tree, analyze feature importance, and perform **cross-validation** for better evaluation.

---

## 📂 Dataset
- The dataset used is **heart.csv**.
- The **target** column represents the presence (1) or absence (0) of heart disease.  
- Features include health metrics such as age, cholesterol level, blood pressure, etc.

---

## 🛠️ Libraries Used
```bash
pandas
numpy
matplotlib
scikit-learn
```

---

## 🚀 Steps in the Code

### 1. **Import Libraries**
We import required libraries for data analysis, modeling, and visualization.

### 2. **Load Dataset**
Load the dataset using `pandas.read_csv()` and check shape, columns, and initial records.

### 3. **Data Preprocessing**
- Separate features (`X`) and target (`y`)  
- Split data into **training** and **testing** sets using `train_test_split`.

### 4. **Decision Tree Classifier**
- Train a **Decision Tree** model using `DecisionTreeClassifier`.  
- Limit `max_depth` to prevent overfitting.  
- Visualize the tree using `plot_tree()`.  

### 5. **Random Forest Classifier**
- Train a **Random Forest** model with `n_estimators=100`.  
- Compare accuracy with Decision Tree results.

### 6. **Feature Importance**
- Extract feature importance scores from Random Forest.  
- Visualize using a bar chart.

### 7. **Cross-Validation**
- Perform **5-fold cross-validation** using `cross_val_score`.  
- Display mean accuracy for model robustness.

---

## 📊 Outputs
1. **Accuracy Scores** for both models  
2. **Classification Reports** with precision, recall, and F1-score  
3. **Decision Tree Visualization**  
4. **Feature Importance Plot**  
5. **Cross-Validation Mean Accuracy**

---

## 🖼️ Example Plots
- Decision Tree Diagram  
- Feature Importance Bar Graph  

---

## ▶️ How to Run
1. Install required libraries:
```bash
pip install pandas numpy matplotlib scikit-learn
```
2. Place your `heart.csv` dataset in the correct path.  
3. Run the Python script:
```bash
python heart_disease_classification.py
```

---

## 📈 Results
- Decision Tree provides simple model visualization.  
- Random Forest often gives **higher accuracy** and **better generalization**.  
- Cross-validation ensures robust performance evaluation.

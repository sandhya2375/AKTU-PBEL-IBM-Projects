# ============================================
# PROJECT 1: Salary Prediction using Ensemble Learning
# IBM SkillsBuild - Google Colab Ready
# ============================================

# Step 1: Libraries install karo
!pip install pandas numpy scikit-learn matplotlib seaborn

# Step 2: Libraries import karo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("✅ Sab libraries import ho gaye!")

# ============================================
# Step 3: Dataset create karo (ya Kaggle se download kar sakte ho)
# ============================================
# Ye ek simple salary dataset hai

np.random.seed(42)

data = {
    'Age': np.random.randint(22, 65, 1000),
    'Experience_Years': np.random.randint(0, 40, 1000),
    'Education_Level': np.random.choice(['Bachelor', 'Master', 'PhD', 'HighSchool'], 1000),
    'Job_Role': np.random.choice(['Developer', 'Manager', 'Analyst', 'Designer'], 1000),
    'Department': np.random.choice(['IT', 'HR', 'Sales', 'Operations'], 1000),
}

# Salary calculate karo (realistic formula)
salaries = []
for i in range(1000):
    base = 30000
    base += data['Age'][i] * 500
    base += data['Experience_Years'][i] * 2000
    
    if data['Education_Level'][i] == 'PhD':
        base += 200000
    elif data['Education_Level'][i] == 'Master':
        base += 100000
    elif data['Education_Level'][i] == 'Bachelor':
        base += 50000
    
    if data['Job_Role'][i] == 'Manager':
        base += 150000
    elif data['Job_Role'][i] == 'Developer':
        base += 100000
    
    # Thoda randomness add karo
    base += np.random.randint(-20000, 20000)
    salaries.append(max(30000, base))

data['Salary'] = salaries

# DataFrame banao
df = pd.DataFrame(data)

print("✅ Dataset ready ho gaya!")
print("\nDataset ka shape:", df.shape)
print("\nPehli 5 rows:")
print(df.head())
print("\nData info:")
print(df.info())
print("\nSalary statistics:")
print(df['Salary'].describe())

# ============================================
# Step 4: DATA PREPROCESSING (Tayyari)
# ============================================
print("\n" + "="*50)
print("PREPROCESSING STEP")
print("="*50)

# Missing values check karo
print("\nMissing values:", df.isnull().sum().sum())

# Categorical columns encode karo
le_education = LabelEncoder()
le_role = LabelEncoder()
le_dept = LabelEncoder()

df['Education_Level_Encoded'] = le_education.fit_transform(df['Education_Level'])
df['Job_Role_Encoded'] = le_role.fit_transform(df['Job_Role'])
df['Department_Encoded'] = le_dept.fit_transform(df['Department'])

print("✅ Categorical variables encode ho gaye!")

# Features aur Target separate karo
X = df[['Age', 'Experience_Years', 'Education_Level_Encoded', 'Job_Role_Encoded', 'Department_Encoded']]
y = df['Salary']

print("\nFeatures (X):", X.shape)
print("Target (y):", y.shape)

# Standardization karo (scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Data scaled ho gaya!")

# ============================================
# Step 5: TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", X_train.shape[0])
print("Testing data size:", X_test.shape[0])

# ============================================
# Step 6: ENSEMBLE MODELS TRAIN KARO
# ============================================
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

# Model 1: Random Forest
print("\n🌲 Random Forest training ho raha hai...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("✅ Random Forest trained!")

# Model 2: Gradient Boosting
print("\n📈 Gradient Boosting training ho raha hai...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)
print("✅ Gradient Boosting trained!")

# ============================================
# Step 7: PREDICTIONS KARO
# ============================================
print("\n" + "="*50)
print("PREDICTIONS & EVALUATION")
print("="*50)

y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

# Ensemble (dono models ka average)
y_pred_ensemble = (y_pred_rf + y_pred_gb) / 2

print("\n✅ Predictions complete!")

# ============================================
# Step 8: EVALUATION METRICS
# ============================================

def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name}:")
    print(f"  RMSE (Root Mean Squared Error): ₹{rmse:.2f}")
    print(f"  MAE (Mean Absolute Error):      ₹{mae:.2f}")
    print(f"  R² Score:                       {r2:.4f}")
    return rmse, mae, r2

evaluate_model(y_test, y_pred_rf, "🌲 Random Forest")
evaluate_model(y_test, y_pred_gb, "📈 Gradient Boosting")
evaluate_model(y_test, y_pred_ensemble, "🎯 Ensemble (Combined)")

# ============================================
# Step 9: VISUALIZATION
# ============================================
print("\n" + "="*50)
print("GRAPHS BANATE HAIN...")
print("="*50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Actual vs Predicted (Random Forest)
axes[0, 0].scatter(y_test, y_pred_rf, alpha=0.5, color='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Salary')
axes[0, 0].set_ylabel('Predicted Salary')
axes[0, 0].set_title('Random Forest: Actual vs Predicted')
axes[0, 0].grid(True, alpha=0.3)

# Actual vs Predicted (Gradient Boosting)
axes[0, 1].scatter(y_test, y_pred_gb, alpha=0.5, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Salary')
axes[0, 1].set_ylabel('Predicted Salary')
axes[0, 1].set_title('Gradient Boosting: Actual vs Predicted')
axes[0, 1].grid(True, alpha=0.3)

# Residuals (Random Forest)
residuals_rf = y_test - y_pred_rf
axes[1, 0].scatter(y_pred_rf, residuals_rf, alpha=0.5, color='blue')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Salary')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Random Forest: Residuals')
axes[1, 0].grid(True, alpha=0.3)

# Residuals (Gradient Boosting)
residuals_gb = y_test - y_pred_gb
axes[1, 1].scatter(y_pred_gb, residuals_gb, alpha=0.5, color='green')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Salary')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Gradient Boosting: Residuals')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Graphs ready!")

# ============================================
# Step 10: PREDICTION KARO NEW DATA PE
# ============================================
print("\n" + "="*50)
print("SAMPLE PREDICTION")
print("="*50)

# Ek naya candidate ka salary predict karo
new_candidate = np.array([[35, 10, 1, 0, 2]])  # [Age, Experience, Education, Role, Dept]
new_candidate_scaled = scaler.transform(new_candidate)

pred_salary_ensemble = y_pred_ensemble_new = (
    rf_model.predict(new_candidate_scaled)[0] + 
    gb_model.predict(new_candidate_scaled)[0]
) / 2

print("\n🎯 Sample Prediction:")
print(f"Age: 35, Experience: 10 years, Education: Master, Role: Developer")
print(f"Predicted Salary: ₹{pred_salary_ensemble:.2f}")

print("\n" + "="*50)
print("✅ PROJECT COMPLETE!")
print("="*50)

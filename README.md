This project utilizes a CatBoost classifier to predict PET (presumably a measure related to heat stress) based on several features. The classifier is optimized using Bayesian Optimization, and SHAP (SHapley Additive exPlanations) values are used to interpret the model's predictions.

Prerequisites
Make sure you have the following libraries installed:

bash
Copy code
pip install pandas scikit-learn catboost bayesian-optimization shap matplotlib
Data
The dataset should be in CSV format located at F:\Research\Data science & Urban geometry\Results\ML\PET.csv. Ensure your data is formatted similarly or update the file_path variable in the script accordingly.

Features
The following features are used to predict the target variable PET:

OR
H/W
H
NT
W
Project Structure
Data Loading: Loading the dataset using pandas.
Feature Engineering: Creating dummy variables for categorical features.
Train-Test Split: Splitting the data into training and testing sets.
Bayesian Optimization: Finding the best hyperparameters for the CatBoost classifier.
Model Training and Evaluation: Training the model with the best parameters and evaluating it.
Model Interpretation with SHAP: Interpreting the model's predictions using SHAP values.
Saving Plots: Saving SHAP plots and permutation feature importance plots.
Code Overview
1. Loading Data
python
Copy code
import pandas as pd

file_path = r"F:\Research\Data science & Urban geometry\Results\ML\PET.csv"
data = pd.read_csv(file_path)
2. Feature Engineering
python
Copy code
feature_columns = ['OR', 'H/W', 'H', 'NT', 'W']
features = data[feature_columns]
target = data['PET']
features_with_dummies = pd.get_dummies(features)
3. Train-Test Split
python
Copy code
from sklearn.model_selection import train_test_split

features_train, features_test, target_train, target_test = train_test_split(
    features_with_dummies, target, test_size=0.3, random_state=2)
4. Bayesian Optimization
python
Copy code
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization

def catboost_classifier(depth, learning_rate, iterations, l2_leaf_reg):
    clf = CatBoostClassifier(
        depth=int(depth),
        learning_rate=learning_rate,
        iterations=int(iterations),
        l2_leaf_reg=l2_leaf_reg,
        random_state=42,
        verbose=False
    )
    clf.fit(features_train, target_train, verbose=False)
    return clf.score(features_test, target_test)

pbounds = {'depth': (4, 10), 'learning_rate': (0.01, 0.3), 'iterations': (50, 200), 'l2_leaf_reg': (1, 10)}

optimizer = BayesianOptimization(f=catboost_classifier, pbounds=pbounds, verbose=2, random_state=0)
optimizer.maximize(init_points=5, n_iter=10)

best_params = optimizer.max['params']
5. Model Training and Evaluation
python
Copy code
from sklearn.metrics import classification_report

best_cb_model = CatBoostClassifier(
    depth=int(best_params['depth']),
    learning_rate=best_params['learning_rate'],
    iterations=int(best_params['iterations']),
    l2_leaf_reg=best_params['l2_leaf_reg'],
    random_state=42,
    verbose=False
)
best_cb_model.fit(features_train, target_train, verbose=False)
y_pred = best_cb_model.predict(features_test)
print(classification_report(target_test, y_pred))
6. Model Interpretation with SHAP
python
Copy code
import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(best_cb_model)
shap_values = explainer.shap_values(features_train)

class_names = ['Moderate heat stress', 'Strong heat stress', 'Extreme heat stress']
shap.summary_plot(shap_values, features_train, plot_type="bar", class_names=class_names)
7. Saving Plots
python
Copy code
save_path_summary = r"F:\Research\Data science & Urban geometry\Results\ML\CATBOOST\Catboost_PET\shap_summary_plot.png"
plt.savefig(save_path_summary)
plt.close()

for i, target in enumerate(class_names):
    shap.summary_plot(shap_values[i], features_train, show=False, plot_type="bar")
    plt.title(f"Parameter Importance for {target}")
    save_path_param_importance = f"F:\Research\Data science & Urban geometry\Results\ML\CATBOOST\Catboost_PET\Parameter_Importance_{target}.jpg"
    plt.savefig(save_path_param_importance, dpi=300, bbox_inches='tight')
    plt.close()

    shap.summary_plot(shap_values[i], features_train, show=False)
    plt.title(f"Parameter Influence for {target}")
    save_path_param_influence = f"F:\Research\Data science & Urban geometry\Results\ML\CATBOOST\Catboost_PET\Parameter_Influence_{target}.jpg"
    plt.savefig(save_path_param_influence, dpi=300, bbox_inches='tight')
    plt.close()

# Replace perm_importance with your actual calculation of permutation feature importance
plt.bar(features_test.columns, perm_importance)
plt.title("Permutation Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha="right")
save_path_perm_importance = r"F:\Research\Data science & Urban geometry\Results\ML\CATBOOST\Catboost_PET\Permutation_Feature_Importance.jpg"
plt.savefig(save_path_perm_importance, dpi=300, bbox_inches='tight')
plt.close()

shap.summary_plot(shap_values, features_train, plot_type="bar", class_names=class_names, show=False)
plt.title("Mean Value SHAP Plot")
save_path_mean_shap = r"F:\Research\Data science & Urban geometry\Results\ML\CATBOOST\Catboost_PET\Mean_Value_SHAP_Plot.jpg"
plt.savefig(save_path_mean_shap, dpi=300, bbox_inches='tight')
plt.close()
Author
This code was created by Alireza Karimi. If you have any questions or suggestions, please contact me at alikar1@alum.us.es 

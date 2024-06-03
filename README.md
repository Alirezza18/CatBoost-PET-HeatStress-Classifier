 import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from bayes_opt import BayesianOptimization

file_path = r"F:\Research\Data science & Urban geometry\Results\ML\PET.csv"  # Updated file path
data = pd.read_csv(file_path)

feature_columns = ['OR', 'H/W', 'H', 'NT', 'W']
features = data[feature_columns]
target = data['PET']

features_with_dummies = pd.get_dummies(features)
features_train, features_test, target_train, target_test = train_test_split(features_with_dummies, target, test_size=0.3, random_state=2)

def catboost_classifier(depth, learning_rate, iterations, l2_leaf_reg):
    clf = CatBoostClassifier(depth=int(depth),
                             learning_rate=learning_rate,
                             iterations=int(iterations),
                             l2_leaf_reg=l2_leaf_reg,
                             random_state=42,
                             verbose=False)
    clf.fit(features_train, target_train, verbose=False)
    return clf.score(features_test, target_test)

# Define parameter bounds
pbounds = {'depth': (4, 10),
           'learning_rate': (0.01, 0.3),
           'iterations': (50, 200),
           'l2_leaf_reg': (1, 10)}

# Perform Bayesian Optimization
optimizer = BayesianOptimization(
    f=catboost_classifier,
    pbounds=pbounds,
    verbose=2,
    random_state=0,
)
optimizer.maximize(init_points=5, n_iter=10)

best_params = optimizer.max['params']
print("Best parameters:", best_params)

# Train CatBoost with the best parameters
best_cb_model = CatBoostClassifier(depth=int(best_params['depth']),
                                   learning_rate=best_params['learning_rate'],
                                   iterations=int(best_params['iterations']),
                                   l2_leaf_reg=best_params['l2_leaf_reg'],
                                   random_state=42,
                                   verbose=False)
best_cb_model.fit(features_train, target_train, verbose=False)
y_pred = best_cb_model.predict(features_test)
print(classification_report(target_test, y_pred))
import shap
import matplotlib.pyplot as plt

# Calculate SHAP values
explainer = shap.TreeExplainer(best_cb_model)
shap_values = explainer.shap_values(features_train)

# Define class names
class_names = ['Moderate heat stress', 'Strong heat stress', 'Extreme heat stress']

# Visualize SHAP summary plot
shap.summary_plot(shap_values, features_train, plot_type="bar", class_names=class_names)

# Save the summary plot
save_path_summary = r"F:\Research\Data science & Urban geometry\Results\ML\CATBOOST\Catboost_PET\shap_summary_plot.png"
plt.savefig(save_path_summary)
plt.close()  # Close the plot to release resources

# Create SHAP summary plots for each target separately
for i, target in enumerate(class_names):
    # Parameter Importance
    shap.summary_plot(shap_values[i], features_train, show=False, plot_type="bar")
    plt.title(f"Parameter Importance for {target}")
    save_path_param_importance = f"F:\Research\Data science & Urban geometry\Results\ML\CATBOOST\Catboost_PET\Parameter_Importance_{target}.jpg"
    plt.savefig(save_path_param_importance, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to release resources

    # Parameter Influence
    shap.summary_plot(shap_values[i], features_train, show=False)
    plt.title(f"Parameter Influence for {target}")
    save_path_param_influence = f"F:\Research\Data science & Urban geometry\Results\ML\CATBOOST\Catboost_PET\Parameter_Influence_{target}.jpg"
    plt.savefig(save_path_param_influence, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to release resources

# Plot Permutation Feature Importance (assuming perm_importance is defined elsewhere)
# Replace perm_importance with your actual calculation of permutation feature importance
plt.bar(features_test.columns, perm_importance)
plt.title("Permutation Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha="right")
save_path_perm_importance = r"F:\Research\Data science & Urban geometry\Results\ML\CATBOOST\Catboost_PET\Permutation_Feature_Importance.jpg"
plt.savefig(save_path_perm_importance, dpi=300, bbox_inches='tight')
plt.close()  # Close the plot to release resources

# Save mean value SHAP plot
shap.summary_plot(shap_values, features_train, plot_type="bar", class_names=class_names, show=False)
plt.title("Mean Value SHAP Plot")
save_path_mean_shap = r"F:\Research\Data science & Urban geometry\Results\ML\CATBOOST\Catboost_PET\Mean_Value_SHAP_Plot.jpg"
plt.savefig(save_path_mean_shap, dpi=300, bbox_inches='tight')
plt.close()  # Close the plot to release resources
This code was created by Alireza Karimi. If you have any questions or suggestions, please contact me at alikar1@alum.us.es 

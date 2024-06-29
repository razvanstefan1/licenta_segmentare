# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import warnings
#
# warnings.filterwarnings('ignore')
# # Load the Titanic dataset
# dataset_file_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\GUI\\RandomForest\\Text_labels_and_Numerical_Data_no_surface_areas_csv.csv'
# eye_measurements = pd.read_csv(dataset_file_path)
# print(eye_measurements.columns)
#
#
# # Drop rows with missing target values
# eye_measurements = eye_measurements.dropna(subset=['Disease'])
#
# # Create a binary target variable for DR
# eye_measurements['DR'] = eye_measurements['Disease'].apply(lambda x: 1 if x == 'DR' else 0)
#
# # Select relevant features and target variable
# X = eye_measurements[['Age', 'FAZ_circularity','Art_density', 'Cap_density','LV_density','Vein_density','Artery_diameter_index','Capillary_diameter_index','Vein_diameter_index','LargeVessel_diameter_index' ]]
#
# #Target variable (variabila ce vrem s o determinam)
# y = eye_measurements['DR']
#
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
# # Create a Random Forest Classifier with 100 trees (estimators) and a random seed of 42
# rf_classifier = RandomForestClassifier(n_estimators=10000, random_state=12, class_weight={0: 1, 1: 10})
#
# # Train the classifier
# rf_classifier.fit(X_train, y_train)
#
# # Make predictions on the test set
# y_pred = rf_classifier.predict(X_test)
#
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred, target_names=['Not DR', 'DR'])
#
# # Print the results
# print(f"Accuracy: {accuracy:.2f}")
# print("\nClassification Report:\n", classification_rep)
import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import warnings
#
# warnings.filterwarnings('ignore')
# # Load the Titanic dataset
# dataset_file_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\GUI\\RandomForest\\Text_labels_and_Numerical_Data_no_surface_areas_csv.csv'
# eye_measurements = pd.read_csv(dataset_file_path)
# print(eye_measurements.columns)
#
#
# # Drop rows with missing target values
# eye_measurements = eye_measurements.dropna(subset=['Disease'])
#
# # Create a binary target variable for DR
# eye_measurements['AMD'] = eye_measurements['Disease'].apply(lambda x: 1 if x == 'AMD' else 0)
#
# # Select relevant features and target variable
# X = eye_measurements[['Age', 'FAZ_circularity','Art_density', 'Cap_density','LV_density','Vein_density','Artery_diameter_index','Capillary_diameter_index','Vein_diameter_index','LargeVessel_diameter_index' ]]
#
# #Target variable (variabila ce vrem s o determinam)
# y = eye_measurements['AMD']
#
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
# # Create a Random Forest Classifier with 100 trees (estimators) and a random seed of 42
# rf_classifier = RandomForestClassifier(n_estimators=5000, random_state=2)
#
# # Train the classifier
# rf_classifier.fit(X_train, y_train)
#
# # Make predictions on the test set
# y_pred = rf_classifier.predict(X_test)
#
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred, target_names=['Not AMD', 'AMD'])
#
# # Print the results
# print(f"Accuracy: {accuracy:.2f}")
# print("\nClassification Report:\n", classification_rep)






import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')
# Load the Titanic dataset
dataset_file_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\GUI\\RandomForest\\Text_labels_and_Numerical_Data_no_surface_areas_csv.csv'
eye_measurements = pd.read_csv(dataset_file_path)
print(eye_measurements.columns)


# Drop rows with missing target values
eye_measurements = eye_measurements.dropna(subset=['Disease'])

# Create a binary target variable for DR
eye_measurements['NORMAL'] = eye_measurements['Disease'].apply(lambda x: 1 if x == 'NORMAL' else 0)

# Select relevant features and target variable
features = ['Age', 'FAZ_circularity', 'Art_density', 'Cap_density', 'LV_density', 'Vein_density', 'Artery_diameter_index', 'Capillary_diameter_index', 'Vein_diameter_index', 'LargeVessel_diameter_index']
X = eye_measurements[features]
#Target variable (variabila ce vrem s o determinam)
y = eye_measurements['NORMAL']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# Create a Random Forest Classifier with 100 trees (estimators) and a random seed of 42
rf_classifier = RandomForestClassifier(n_estimators=5000, random_state=123)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=['Not healthy', 'healthy'])

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)



#GENEREAZA GRAFICUL CU Importanta features-urilor
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

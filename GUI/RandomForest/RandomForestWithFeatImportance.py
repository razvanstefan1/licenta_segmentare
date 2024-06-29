import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz


# Load your data (replace with actual dataset loading)
dataset_file_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\GUI\\RandomForest\\Text_labels_and_Numerical_Data_no_surface_areas_csv.csv'
eye_measurements = pd.read_csv(dataset_file_path)

# Preprocess data
eye_measurements = eye_measurements.dropna(subset=['Disease'])
eye_measurements['DR'] = eye_measurements['Disease'].apply(lambda x: 1 if x == 'DR' else 0)
features = ['Age', 'FAZ_circularity', 'Art_density', 'Cap_density', 'LV_density', 'Vein_density', 'Artery_diameter_index', 'Capillary_diameter_index', 'Vein_diameter_index', 'LargeVessel_diameter_index']
X = eye_measurements[features]
y = eye_measurements['DR']

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=31245)
rf_classifier.fit(X, y)

# Get feature importances
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()



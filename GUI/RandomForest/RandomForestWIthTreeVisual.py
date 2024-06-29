import pandas as pd
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

# Load the dataset
dataset_file_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\GUI\\RandomForest\\Text_labels_and_Numerical_Data_no_surface_areas_csv.csv'
eye_measurements = pd.read_csv(dataset_file_path)

# Preprocess data
eye_measurements = eye_measurements.dropna(subset=['Disease'])
eye_measurements['DR'] = eye_measurements['Disease'].apply(lambda x: 1 if x == 'DR' else 0)
features = ['Age', 'FAZ_circularity', 'Art_density', 'Cap_density', 'LV_density', 'Vein_density', 'Artery_diameter_index', 'Capillary_diameter_index', 'Vein_diameter_index', 'LargeVessel_diameter_index']
X = eye_measurements[features]
y = eye_measurements['DR']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_classifier.fit(X_train, y_train)

# Export the first tree in the forest
tree = rf_classifier.estimators_[0]
export_graphviz(tree, out_file='tree.dot', feature_names=features, class_names=['Not DR', 'DR'], filled=True)

# Read and visualize the tree from the dot file
with open('tree.dot') as f:
    dot_graph = f.read()

graph = graphviz.Source(dot_graph)
graph.render('tree')  # This will save the visualization as 'tree.png'
graph.view()  # This will open the visualization if a viewer is available

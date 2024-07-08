import graphviz
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import numpy as np
from sklearn.tree import export_graphviz

#conceput pentru clasificare binara
#e tot fisier care poate fi rulat de sine statator dar metodele lui pot fi apelate si din alte fisiere (ca si GenerateSegmentation.py)

def GenerateTrainTestRForest(csv_path, target_column, target_disease, feature_list, estimator_count, seed_number, percentage_test_data, show_importance_graph=False, show_tree_visualization=False):
    #citim csv ul
    eye_measurements = pd.read_csv(csv_path)
    print(eye_measurements.columns)

    #scoatem coloana care contine targetul
    eye_measurements = eye_measurements.dropna(subset=[target_column])

    #alegem boala pe care facem clasificarea binara (randurile cu aceasta boala iau valoarea 1, restul 0)
    eye_measurements[target_disease] = eye_measurements[target_column].apply(lambda x: 1 if x == target_disease else 0)

    #alegem feature-urile
    X=eye_measurements[feature_list]

    #boala target
    y=eye_measurements[target_disease]

    #convertim sexul in valori numerice
   # X.loc[:, 'Sex'] = X['Sex'].map({'F': 0, 'M': 1})

    #splittuim datasetul in train si test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percentage_test_data, random_state=seed_number)

    #creem un random forest classifier
    random_forest_classifier = RandomForestClassifier(n_estimators=estimator_count, random_state=seed_number)

    #antrenam clasificatorul
    random_forest_classifier.fit(X_train, y_train)

    #facem predictii pe setul de test
    y_pred = random_forest_classifier.predict(X_test)

    #evaluare model
    accuracy = accuracy_score(y_test, y_pred)
    metrics_report = classification_report(y_test, y_pred, target_names=[f'Not {target_disease}', f'{target_disease}'])

    #printare
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", metrics_report)

    #generare grafic cu importanta feature-urilor
    if show_importance_graph:
        generateFeatureImportanceGraph(random_forest_classifier, feature_list, X)

    #generare vizualizare arbore
    if show_tree_visualization:
        generateTreeVisualization(random_forest_classifier, feature_list, [f'Not {target_disease}', f'{target_disease}'])

    return random_forest_classifier


def generateFeatureImportanceGraph(RF, feature_list, X):
    importances = RF.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_list[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

def generateTreeVisualization(RF, feature_list, target_names_list):
    # GENERAM O vizualizare a primului arbore din forest (pentru a intelege cum se iau deciziile). se genereaza fisier dot si pdf
    tree = RF.estimators_[0] #etragem primul arbore din forest
    export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, class_names=target_names_list, filled=True)

    # citim fisierul dot
    with open('tree.dot') as f:
        dot_graph = f.read()

    graph = graphviz.Source(dot_graph)
    graph.render('tree')
    graph.view()


def GenerateClassificationResult(value_row, feature_list, target_disease, RF):
    #creem un dataframe cu valorile care vor fi clasificate
    property_data_frame = pd.DataFrame([value_row], columns=feature_list)

    #facem predictia
    prediction = RF.predict(property_data_frame)

    #printam rezultatul predictiei
    if prediction[0] == 1:
        print(f'Classification result: {target_disease}')
    else:
        print(f'Classification result: NOT {target_disease}')

    return prediction[0]

if __name__ == '__main__':
    csv_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\GUI\\RandomForest\\Text_labels_and_Numerical_Data_no_surface_areas_csv.csv'
    #csv_path = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\GUI\\RandomForest\\Text_labels_and_Numerical_Data_plusGender.csv'
    target_column = 'Disease'
    target_disease = 'NORMAL'
    feature_list = ['Age', 'FAZ_circularity', 'Art_density', 'Cap_density', 'LV_density', 'Vein_density', 'Artery_diameter_index', 'Capillary_diameter_index', 'Vein_diameter_index', 'LargeVessel_diameter_index']
    #feature_list = [ 'Age', 'Sex', 'FAZ_circularity', 'Art_density', 'Cap_density', 'Vein_density', 'Artery_diameter_index', 'Capillary_diameter_index', 'Vein_diameter_index', 'LargeVessel_diameter_index']
    estimator_count = 5000
    seed_number = 123
    percentage_test_data = 0.2

    RF=GenerateTrainTestRForest(csv_path, target_column, target_disease, feature_list, estimator_count, seed_number, percentage_test_data, show_importance_graph=True, show_tree_visualization=True)

    joblib.dump(RF, 'random_forest_model.pkl')
    #print('generare')
    #GenerateClassificationResult([55,0.459722275,0.0390625,0.46563125,0.08276875,0.043975,8.933119744,6.331678888,8.935320228,8.899184149], feature_list, target_disease, RF)

# import statements
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from joblib import dump
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# import datat and separate labels
data = pd.read_csv('../../Data/Curated/UseCaseDataModeling.csv')

Cases = data['Harmonized Functional Use'].unique()

ModelsDir = '../../Modeling/IsoForest/IsoForestModels'

os.makedirs(ModelsDir, exist_ok=True)


holder = {}

# File to save the metrics
metrics_file = '/home/matt/Proj/QSURv3/Results/FFTrainingResults.csv'
with open(metrics_file, 'w') as mf:
    mf.write("Use_Case,Accuracy,Precision,Recall,F1_Score,AUC_Score\n")




for UseCase in Cases:
    try:

        df = data[data['Harmonized Functional Use']==UseCase]
        y = df['Harmonized Functional Use']
        X= df.filter(regex='^Bit')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        iso_forest = IsolationForest(n_estimators=100, random_state=42)
        iso_forest.fit(X_train)
        UseCase=UseCase.replace('/','_')
        model_path = os.path.join(ModelsDir, f'{UseCase}.joblib')


        dump(iso_forest, model_path)

        print(f'Model for {UseCase} complete')



        scores = iso_forest.decision_function(X_test)
        # Invert scores because lower scores mean more anomalous in Isolation Forest


        # Use the median score as the threshold
        threshold = np.median(scores)

        predictions = np.where(scores >= threshold, 1, 0)


            # True labels are all 1s since the test set only contains positive examples
        true_labels = np.ones(len(predictions))

        # Calculate precision, recall, and F1 score
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)


            # Append metrics to the file
        with open(metrics_file, 'a') as mf:
            mf.write(f"{UseCase},'Precision:'{precision},'Recall:'{recall},F1:{f1}\n")

    
        traindata =  list(iso_forest.decision_function(X_train))
        testdata=list(iso_forest.decision_function(X_test))
        

        TrainTest =  {UseCase:{'Train': traindata, 'Test': testdata}}

        holder.update(TrainTest) 



        print(f"Model and metrics for {UseCase} saved.")
    except:
        next
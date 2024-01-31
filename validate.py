# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:17:44 2024

@author: lika
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from joblib import load

# загрузка модели и векторизатора 
antivirus = load('antivirus.joblib')
mlb = load( 'mlb.joblib')

# загрузка и предобработка данных
val_data = pd.read_csv("val.tsv", sep="\t")
val_libs = [libs.split(',') for libs in val_data['libs']]
x_val = mlb.transform(val_libs)
y_val = val_data['is_virus']

# оценка точности на валидационных данных
y_pred = antivirus.predict(x_val)

# расчет матрицы ошибок
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# запись результатов в файл
with open("validation.txt", "w") as file:
    file.write(f"True positive: {tp}\n")
    file.write(f"False positive: {fp}\n")
    file.write(f"False negative: {fn}\n")
    file.write(f"True negative: {tn}\n")
    file.write(f"Accuracy: {accuracy:.2f}\n")
    file.write(f"Precision: {precision:.2f}\n")
    file.write(f"Recall: {recall:.2f}\n")
    file.write(f"F1: {f1:.2f}\n")
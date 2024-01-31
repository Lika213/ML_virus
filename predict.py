# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:37:52 2024

@author: lika
"""
import pandas as pd
from joblib import load
import numpy as np

# загрузка модели и векторизатора 
antivirus = load('antivirus.joblib')
mlb = load( 'mlb.joblib')

# загрузка и предобработка данных
test_data = pd.read_csv("test.tsv", sep="\t")
test_libs = [libs.split(',') for libs in test_data['libs']]
x_test = mlb.transform(test_libs)


# предсказание классов файлов
y_pred = antivirus.predict(x_test)

# запись предсказаний в файл
with open("prediction.txt", "w") as file:
    file.write("prediction\n")
    for pred in y_pred:
        file.write(f"{pred}\n")
        

# получаем коэффициенты модели для анализа вклада признаков
coefficients = antivirus.coef_[0]

# получаем имена библиотек из векторизатора
libs_names = mlb.classes_

with open("explain.txt", "w") as explain_file:
    explain_file.write("prediction\n")
    # запускаем цикл, проходящий по всем предсказаниям 
    for idx, pred in enumerate(y_pred):
        if pred == 1:
            # для зловредных файлов находим признаки с наибольшим положительным вкладом
            top_indices = np.argsort(coefficients)[::-1][:5]  # Топ-5 признаков
            top_features = [libs_names[i] for i in top_indices if coefficients[i] > 0]
            explain_file.write("Классифицирован как зловредный из-за наличия библиотек: " + ", ".join(top_features) + "\n")
        else:
            # для безопасных файлов
            explain_file.write("\n")  

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:58:07 2024

@author: lika
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from joblib import dump

# загрузка и преобработка обучающих данных
train_data = pd.read_csv("train.tsv", sep="\t")
train_libs = [libs.split(',') for libs in train_data['libs']]

# инициализация и обучение MultiLabelBinarizer для векторизации списка библиотек
mlb = MultiLabelBinarizer()
x_train = mlb.fit_transform(train_libs)

# получение целевой переменной
y_train = train_data['is_virus']

# обучение модели логистической регрессии
antivirus = LogisticRegression(C = 0.72, max_iter = 500)
antivirus.fit(x_train, y_train)

# сохранение модели и векторизатора
dump(antivirus, 'antivirus.joblib')
dump(mlb, 'mlb.joblib')





import numpy as np
# data processing
import pandas as pd
# visualization
import matplotlib.pyplot as plt
import seaborn as sns
# machine learning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Process 1 : Load the Data
data_df =pd.read_csv('TitanicDataset\\titanicpassengers.csv')
# distribution of numerical features
print(data_df.describe())

#Step 2: Preprocessing the data
'''
#Using For loop convert the values
# convert Sex categorical feature to numerical.
for df in all_df:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
'''

# Step 3: Create Pivot
# sex of passengers
pivot1 = data_df[['Gender', 'Survived']].groupby(['Gender']).mean().sort_values(by='Survived', ascending=False)
print(pivot1)
# class of passengers
pivot1 = data_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(pivot1)
# sibling and spouse
pivot1 = data_df[['sibsp', 'Survived']].groupby(['sibsp']).mean().sort_values(by='Survived', ascending=False)
print(pivot1)
# parents and children
pivot1 = data_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(pivot1)
# embark location
pivot1 = data_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(pivot1)

#Divide the dataset into train and test set

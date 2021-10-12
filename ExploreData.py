import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer


dataTrain = pd.read_csv('train.csv')

#Récupérer les colonnes Pclass, Sex, Age Sibsp, Parch, Fare,Embareked

param = dataTrain.iloc[:,[2,4,5,6,7,9,11,1]].values
result = dataTrain.iloc[:,1].values


#Traitement des données manquantes

imputer = SimpleImputer()
imputer = imputer.fit(param[:,[0,2,3,4,5]])
param[:, [0,2,3,4,5]] = imputer.transform(param[:, [0,2,3,4,5]])
param[:,2] = param[:,2].astype(int)

#Encodage des variables catégoriques:
    
labelencoder = LabelEncoder()
param[:,1] = labelencoder.fit_transform(param[:,1]) #Cette ligne permet de numériser les données de la colonne SEX
param[:,-1] = labelencoder.fit_transform(param[:,-1])
param = pd.DataFrame(data=param, columns=["Pclass", "Sex", "Age", "Sibsp", "Parch", "Fare", "Embareked","Survived"])

'''ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [5])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)

param = ct.fit_transform(param)

param = param.toarray()'''

# Analyse de données


occurence = param.iloc[:,[0,-1,2]]
occurence = occurence.value_counts()


occurence = occurence.reset_index()
#occurence["occurence"] = occurence[0]


occurence = occurence.sort_values(by = ['Pclass','Survived'])
occurence = occurence.to_numpy()
print(occurence)
occurence = np.reshape(occurence, (3,2,3))

occurence = occurence[:,:,2].astype(float)
for i in range(len(occurence)):
    occurence[i] = occurence[i] / sum(occurence[i]) *100
print(occurence)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x = np.arange(3) 
y = occurence[:,1]
yprim = occurence[:,0]
ax.bar(x,y, color = 'b', width = 0.25) #qui ont survécu
ax.bar(x+0.25,yprim, color = 'r', width = 0.25) #qui n'ont survécu
ax.legend(labels=['survivers', 'dead'])
axes = plt.gca()
axes.set_ylim(0, 100)
plt.xticks([r + 0.25 / 2 for r in range(len(x))], ["classe1","classe2", "classe3" ])


plt.show()

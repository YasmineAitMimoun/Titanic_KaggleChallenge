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

occurence = occurence.reset_index()

ages = pd.DataFrame(data = occurence, columns=["Age"])

occurence  = occurence.drop(["Age","index"], axis = 1)

ages = ages.where(ages >= 18, 0) #0 pour enfant

ages = ages.where(ages < 18, 1) #1 pour adulte

classeSurv = occurence

occurence = (pd.concat([occurence, ages],axis = 1))


occurence = occurence.value_counts()
occurence = occurence.reset_index()



occurence = occurence.sort_values(by = ['Pclass','Age','Survived'])
data = occurence.to_numpy()
data = np.reshape(data, (3,4,4))

for i in range(len(data)):
    data[:,:,-1][i] = data[:,:,-1][i]/ sum(data[:,:,-1][i]) * 100
    


data = np.reshape(data, (12,4))
data = pd.DataFrame(data, columns=(['Pclass','Age','Survived','percent']))
data = data.sort_values(by = ['Age','Pclass','Survived'])

data = data.iloc[[0,2,4,6,8,10]]
print(data)
data = data.drop(["Survived"], axis=1)
data = data.sort_values(by = ['Age','Pclass'])

data = data["percent"].to_numpy()
print(data)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x = np.arange(3) 
y = data[:3]
yprim = data[3:]
ax.bar(x,y, color = 'b', width = 0.25) #qui ont survécu
ax.bar(x+0.25,yprim, color = 'r', width = 0.25) #qui n'ont survécu
ax.legend(labels=['Enfants', 'Adultes'])
axes = plt.gca()
plt.xticks([r + 0.25 / 2 for r in range(len(x))], ["classe1","classe2", "classe3" ])


plt.show()


classeSurv = classeSurv.value_counts()
classeSurv = classeSurv.reset_index()

classeSurv = classeSurv.sort_values(by = ['Pclass','Survived'])
classeSurv = classeSurv.to_numpy()
print(classeSurv)
classeSurv = np.reshape(classeSurv, (3,2,3))

classeSurv = classeSurv[:,:,2].astype(float)
for i in range(len(classeSurv)):
    classeSurv[i] = classeSurv[i] / sum(classeSurv[i]) *100
print(classeSurv)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x = np.arange(3) 
y = classeSurv[:,1]
yprim = classeSurv[:,0]
ax.bar(x,yprim+y, color = ['red' for i in y], width = 0.25) #qui n'ont survécu
ax.bar(x,y, color = ['blue' for i in y], width = 0.25) #qui ont survécu
ax.legend(labels=['dead', 'survivers'])
axes = plt.gca()
axes.set_ylim(0, 110)
plt.xticks([r for r in range(len(x))], ["classe1","classe2", "classe3" ])
plt.title("Pourcentage de morts et de survivants en fonction des classes")

plt.show()

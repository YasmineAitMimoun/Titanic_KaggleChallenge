import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

class dataset:
    def __init__(self, dataTrainfile, dataTestFile):
        self.dataTrain = pd.read_csv(dataTrainfile)
        self.dataTest = pd.read_csv(dataTestFile)
        
    def getColumns(self, dataFrame, columns):
        return dataFrame.loc[: ,columns].values
        
    def fillMissingData(self, dataFrame, indexColumns):
        imputer = SimpleImputer()
        imputer = imputer.fit(dataFrame[:,indexColumns])
        dataFrame[:,indexColumns] = imputer.transform(dataFrame[:,indexColumns])
    
    def numerizeData(self, dataFrame, indexColumns):
        labelencoder = LabelEncoder()
        for i in indexColumns:
            dataFrame[:,i] = labelencoder.fit_transform(dataFrame[:,i]) #Cette ligne permet de numériser les données de la colonne i
        
    def encodeData(self,dataFrame, indexColumns):
        ct = ColumnTransformer( 
            [('one_hot_encoder', OneHotEncoder(categories='auto'), [indexColumns])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
            remainder='passthrough')                                         # Leave the rest of the columns untouched
        
        dataFrame = ct.fit_transform(dataFrame)


dataset = dataset("train.csv","train.csv")

ExtractedData = dataset.getColumns(dataset.dataTrain, ['Pclass' ,'Sex' ,'Age' ,'SibSp' ,'Parch' ,'Fare' ,'Embarked'])

dataset.fillMissingData(ExtractedData,[0,2,3,4,5] )

ExtractedData[:,2] = ExtractedData[:,2].astype(int)

dataset.numerizeData(ExtractedData, [1,-1])
ExtractedData = pd.DataFrame(data=ExtractedData, columns=["Pclass", "Sex", "Age", "Sibsp", "Parch", "Fare", "Embareked"])


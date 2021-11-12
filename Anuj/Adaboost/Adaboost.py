import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
import pickle

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils import dataset_summary

class AdaBoost:
    def __init__(self):  
        self.model = AdaBoostClassifier(n_estimators=200, random_state=0)
        
    def train(self, X, y, checkpoint_dir):
        self.model.fit(X, y)
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        
        with open(checkpoint_dir + "checkpoint.chk", 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, checkpoint_dir):
        if not os.path.isfile(checkpoint_dir + "checkpoint.chk"):
            print("No model found!")
            return

        with open(checkpoint_dir + "checkpoint.chk", 'rb') as f:
            self.model = pickle.load(f)

    def evaluate(self, test_X, test_y):
        
        y_pred = self.model.predict(test_X)
        y_pred = y_pred >= 0.5
        
        print()
        print("Classification Report:")
        print(classification_report(test_y, y_pred))
        
        cf_matrix = confusion_matrix(test_y, y_pred)
        
        print()
        print("Confusion Matrix:")
        print(cf_matrix)
        
        sns.heatmap(cf_matrix, annot=True)
        
        plt.show()
        
def read_dataset(csv_file_path, test_size=0.2, random_state=0):
    df = pd.read_csv(csv_file_path)
    df = df.drop(['Amount','Time'], axis=1)
    
    y = df['Class']
    X = df.drop(['Class'], axis=1)
    
    #Resampling
    #over = SMOTE(sampling_strategy=0.05)
    #under = RandomUnderSampler(sampling_strategy=0.25)
    #steps = [('o', over), ('u', under)]
    #pipeline = Pipeline(steps=steps)
    #X,y = pipeline.fit_resample(X,y)
    
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = test_size, random_state=random_state)
    
    #MinMax scaling
    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    
    return [train_X, test_X], [train_y, test_y]
    
if __name__ == '__main__':
    
    X,y = read_dataset('../creditcard.csv')
    dataset_summary(X,y)
    train_X, test_X = X
    train_y, test_y = y
    
    checkpoint_dir = "checkpoints/"
    
    model = AdaBoost()
    model.train(train_X, train_y, checkpoint_dir=checkpoint_dir)
    # model.load(checkpoint_dir)
    model.evaluate(test_X, test_y)
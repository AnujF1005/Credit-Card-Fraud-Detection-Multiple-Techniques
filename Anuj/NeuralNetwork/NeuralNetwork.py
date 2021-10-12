from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils import dataset_summary

class NeuralNetwork:
    def __init__(self, X, y):
    
        self.train_X, self.test_X = X[0], X[1]
        self.train_y, self.test_y = y[0], y[1]
        
        self.model = Sequential()
        self.model.add(Input(shape=(self.train_X.shape[1],)))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    def summary(self):
        print(self.model.summary())
        
    def train(self, batch_size, epochs, checkpoint_path):
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=5*batch_size)
        
        self.model.fit(self.train_X, self.train_y, batch_size=batch_size, epochs=epochs, callbacks=[cp_callback])
    
    def load(self, chk_dir):
        
        weights = tf.train.latest_checkpoint(chk_dir)
        self.model.load_weights(weights)
    
    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.test_X, self.test_y)
        print('Loss:', loss)
        print('Accuracy:', accuracy)
        
        y_pred = self.model.predict(self.test_X)
        y_pred = y_pred >= 0.5
        
        print()
        print("Classification Report:")
        print(classification_report(self.test_y, y_pred))
        
        cf_matrix = confusion_matrix(self.test_y, y_pred)
        
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
    over = SMOTE(sampling_strategy=0.05)
    under = RandomUnderSampler(sampling_strategy=0.25)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X,y = pipeline.fit_resample(X,y)
    
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = test_size, random_state=random_state)
    
    #MinMax scaling
    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    
    return [train_X, test_X], [train_y, test_y]
    
if __name__ == '__main__':
    
    X,y = read_dataset('../creditcard.csv')
    dataset_summary(X,y)
    
    checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    model = NeuralNetwork(X, y)
    #model.train(batch_size=16, epochs=4, checkpoint_path=checkpoint_path)
    model.load(checkpoint_dir)
    model.evaluate()
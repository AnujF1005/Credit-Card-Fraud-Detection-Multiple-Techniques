from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import os
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, X, y, test_size=0.2, random_state=0):
    
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size = test_size, random_state=random_state)
        
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
        cf_matrix = confusion_matrix(self.test_y, y_pred)
        sns.heatmap(cf_matrix, annot=True)
        
        plt.show()
        
def read_dataset(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df = df.drop(['Amount','Time'], axis=1)
    
    y = df['Class']
    X = df.drop(['Class'], axis=1)
    
    return X,y
    
if __name__ == '__main__':
    
    X,y = read_dataset('creditcard.csv')
    
    checkpoint_path = "chekpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    model = NeuralNetwork(X, y, random_state=101)
    #model.train(batch_size=16, epochs=2, checkpoint_path=checkpoint_path)
    model.load(checkpoint_dir)
    model.evaluate()
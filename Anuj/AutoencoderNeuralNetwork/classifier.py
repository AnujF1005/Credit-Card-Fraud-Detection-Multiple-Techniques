from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import datetime

class Classifier():
    def __init__ (self):
        self.model = Sequential()
        self.model.add(Input(shape=(29,)))
        self.model.add(Dense(22, activation='relu'))
        self.model.add(Dense(15, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(5, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def summary(self):
        print(self.model.summary())
    
    def train(self, X, y, batch_size, epochs, checkpoint_path):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq='epoch')
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        self.model.fit(X,y,batch_size=batch_size, epochs=epochs, callbacks=[cp_callback, tensorboard_callback])
        
    def load(self, chk_dir):   
        weights = tf.train.latest_checkpoint(chk_dir)
        self.model.load_weights(weights)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def evaluate(self, X, y):
        loss, accuracy = self.model.evaluate(X, y)
        print('Loss:', loss)
        print('Accuracy:', accuracy)
        
        y_pred = self.model.predict(X)
        y_pred = y_pred >= 0.5
        
        print()
        print("Classification Report:")
        print(classification_report(y, y_pred))
        
        cf_matrix = confusion_matrix(y, y_pred)
        
        print()
        print("Confusion Matrix:")
        print(cf_matrix)
        
        sns.heatmap(cf_matrix, annot=True)
        
        plt.show()
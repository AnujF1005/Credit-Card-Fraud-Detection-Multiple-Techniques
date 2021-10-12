from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.losses import MeanSquaredError

class Autoencoder():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Input(shape=(29,)))
        self.model.add(Dense(22))
        self.model.add(Dense(15))
        self.model.add(Dense(10))
        self.model.add(Dense(15))
        self.model.add(Dense(22))
        self.model.add(Dense(29))
        
        self.model.compile(optimizer='sgd', loss = MeanSquaredError(), metrics=['accuracy'])
        
    def summary(self):
        print(self.model.summary())
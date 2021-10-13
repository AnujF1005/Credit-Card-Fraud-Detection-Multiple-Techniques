import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from autoencoder import Autoencoder
from classifier import Classifier
from utils import dataset_summary
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def read_dataset(csv_file_path, test_size=0.2, random_state=0):
    df = pd.read_csv(csv_file_path)
    df = df.drop(['Time'], axis=1)
    
    #MinMax scaling
    scaler = MinMaxScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    y = df['Class']
    X = df.drop(['Class'], axis=1)
    
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = test_size, random_state=random_state)
    
    #Resampling
    over = SMOTE()
    steps = [('o', over)]
    pipeline = Pipeline(steps=steps)
    train_X,train_y = pipeline.fit_resample(train_X,train_y)
    
    return [train_X, test_X], [train_y, test_y]
    
if __name__ == '__main__':
    
    X,y = read_dataset('../creditcard.csv')
    dataset_summary(X,y)
    
    train_X, test_X = X
    train_y, test_y = y
    
    aenc = Autoencoder()
    noised_X = aenc.add_gausian_noise(train_X)
    
    autoencoder_checkpoint_path = "autoencoder_checkpoints/cp-{epoch:04d}.ckpt"
    autoencoder_checkpoint_dir = os.path.dirname(autoencoder_checkpoint_path)
    
    #aenc.train(train_X, noised_X, batch_size=64, epochs=2, checkpoint_path=autoencoder_checkpoint_path)
    aenc.load(autoencoder_checkpoint_dir)
    
    train_X = aenc.predict(noised_X)
    test_X = aenc.predict(test_X)
    
    classifier_checkpoint_path = "classifier_checkpoints/cp-{epoch:04d}.ckpt"
    classifier_checkpoint_dir = os.path.dirname(classifier_checkpoint_path)
    
    clf = Classifier()
    clf.train(train_X, train_y, batch_size=64, epochs=2, checkpoint_path=classifier_checkpoint_path)
    #clf.load(classifier_checkpoint_dir)
    clf.evaluate(test_X, test_y)
    
    
    
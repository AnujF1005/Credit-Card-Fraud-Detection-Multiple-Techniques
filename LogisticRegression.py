import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import scikitplot as skplt
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import RandomOverSampler

def model_evaluation(Y_test,Y_pred):
	print(classification_report(Y_test,Y_pred))
	print("Accuracy Score: ",accuracy_score(Y_test,Y_pred))
	skplt.metrics.plot_confusion_matrix(Y_test, Y_pred)
	plt.show()
	



df=pd.read_csv("creditcard.csv")
data_features = df.iloc[:, 0:30]
data_targets = df.iloc[:, 30:]
data_features['scaled_amount'] = RobustScaler().fit_transform(data_features['Amount'].values.reshape(-1,1))
data_features['scaled_time'] = RobustScaler().fit_transform(data_features['Time'].values.reshape(-1,1))
data_features_scaled = data_features.drop(['Time','Amount'],axis = 1,inplace=False)
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(data_features_scaled, data_targets)


np.random.seed(42)
X_train, X_test, Y_train, Y_test = train_test_split(X_over, y_over,train_size = 0.70, test_size = 0.30, random_state = 1)
lr = LogisticRegression(penalty="l2", C=5)
lr.fit(X_train,Y_train.values.ravel())
Y_pred=lr.predict(X_test)
model_evaluation(Y_test,Y_pred)

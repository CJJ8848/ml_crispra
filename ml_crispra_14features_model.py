import numpy as np
from os.path import dirname
import joblib
import sys
# #predict with the trained model
user_train = sys.argv[1]

#print(str(user_train)+"fffff")
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#user_train = "-0.18708664, 1.73239376, 1.54802069,  0.7466342,   1.37401474,  0.33645973, 2.11035398,  1.8803562,   0.72672721 , 2.28216878 , 0.56584782,  1.82112032  ,0.88874096 , 0.72263392,AFFFF"
#print(list(map(float,user_train.split(","))))
user_array = np.array(list(user_train.split(",")))

user_array = np.array(list(map(float,user_array[0:14])))

#print(user_array.reshape(1,-1))
## 3) 标准化z-score
#transfer = StandardScaler()
#user_array = transfer.fit_transform(user_array[:, np.newaxis])
estimator_Bag =joblib.load(dirname(__file__) + "/models_of_human/" + "/Bag/" +"my_lr_0.8804113863871038.pkl")
estimator_Ext =joblib.load(dirname(__file__) + "/models_of_human/" + "/Ext/" +"my_lr_0.7987099250805105.pkl")
estimator_Ran =joblib.load(dirname(__file__) + "/models_of_human/" + "/Ran/"+"my_lr_0.8476145449021282.pkl")
estimator_XGB =joblib.load(dirname(__file__) + "/models_of_human/" + "/XGB/"+"my_lr_0.8461697925176008.pkl")

y_predict_Ran = estimator_Ran.predict(user_array.reshape(1,-1))[0]
print(str(y_predict_Ran))
y_predict_XGB = estimator_XGB.predict(user_array.reshape(1,-1))[0]
print(str(y_predict_XGB))
y_predict_Bag = estimator_Bag.predict(user_array.reshape(1,-1))[0]
print(str(y_predict_Bag))
y_predict_Ext = estimator_Ext.predict(user_array.reshape(1,-1))[0]
print(str(y_predict_Ext))
y_predict = [y_predict_Ran,y_predict_XGB,y_predict_Bag,y_predict_Ext]
print(sum(y_predict)/len(y_predict))
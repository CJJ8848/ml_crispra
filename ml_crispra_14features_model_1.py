import numpy as np
from os.path import dirname
import joblib
import sys
# #predict with the trained model
user_train = sys.argv[1]
#print(str(user_train)+"fffff")
#user_train = "-0.18708664, 1.73239376, 1.54802069,  0.7466342,   1.37401474,  0.33645973, 2.11035398,  1.8803562,   0.72672721 , 2.28216878 , 0.56584782,  1.82112032  ,0.88874096 , 0.72263392"
user_array = np.array(list(map(float,user_train.split(","))))
estimator_Gra =joblib.load("/Users/cuijiajun/Desktop/SRTP/python/ml_crispra_v4" + "/models_of_human/" + "/Gra/" +"my_lr_0.8309714427430532.pkl")

y_predict_Gra = estimator_Gra.predict(user_array.reshape(1,-1))[0]
print(str(y_predict_Gra))

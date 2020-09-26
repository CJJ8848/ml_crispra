# import csv
# from os.path import join, dirname
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from sklearn.linear_model import Lasso
# from sklearn.neural_network import MLPRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.tree import ExtraTreeRegressor
# from xgboost import XGBRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import BaggingRegressor
# from sklearn.metrics import mean_squared_error
# import joblib
# from sklearn.utils import Bunch
# from sklearn.preprocessing import MinMaxScaler
# import operator
# #scoretxt = 'max_scores_v4_sixmdoels.txt'
# scoretxt = 'dt_predict_0_391.txt'
# #scoretxt = 'gc_abnormaldone_max_scores.txt'
#
# def grabTree(filename):
#     fr = open(filename, 'rb')
#     return joblib.load(fr)
# def load_crispra(*, return_X_y=False):
#     module_path = dirname(__file__)
#
#     #data_file_name = join(module_path, 'data', 'crispra_v3_gc.csv')
#     #scores all positive
#     #data_file_name = join(module_path, 'data', 'crispra_v3_gc_allpositive_0-0.01_Rstudio_log10_14features.csv')
#     data_file_name = join(module_path, 'data', 'crispra_v3_gc_allpositive_0-0.01_Rstudio_log10_features_7.csv')
#     with open(data_file_name) as f:
#         data_file = csv.reader(f)
#         temp =next(data_file)
#         n_samples = int(temp[0])
#         n_features = int(temp[1])
#         data = np.empty((n_samples, n_features))
#         target = np.empty((n_samples,))
#         temp = next(data_file)  # names of features
#         feature_names = np.array(temp)
#
#         for i, d in enumerate(data_file):
#             data[i] = np.asarray(d[:-1], dtype=np.float64)
#             target[i] = np.asarray(d[-1], dtype=np.float64)
#         # #data normalization
#         # minmaxtransfer = MinMaxScaler()
#         # data = minmaxtransfer.fit_transform(data)
#     if return_X_y:
#         return data, target
#
#     return Bunch(data=data,
#                  target=target,
#                  # last column is target value
#                  feature_names=feature_names[:-1],
#                  filename=data_file_name)
# def linear_normalized(model,test_size):
#     # 1） 获取数据
#     crispra = load_crispra()
#     #print("crispra: \n", crispra)
#     #print("feature number: \n",crispra.data.shape)
#     x_train, x_test, y_train, y_test = train_test_split(crispra.data, crispra.target, test_size=test_size)
#     # 2)
#     # 3) 标准化z-score
#     transfer = StandardScaler()
#     x_train = transfer.fit_transform(x_train)
#     x_test = transfer.transform(x_test)
#     # # data normalization
#     # minmaxtransfer = MinMaxScaler()
#     # x_train = minmaxtransfer.fit_transform(x_train)
#     # x_test = minmaxtransfer.fit_transform(x_test)
#     # 加载模型
#     #estimator = joblib.load("my_lr.pkl")
#     # 4) 预估器
#     estimator=model
#     estimator.fit(x_train,y_train)
#     # 5)得出model
#     #print("正规方程权重系数为：\n",estimator.coef_)
#     #print("正规方程偏置为：\n", estimator.intercept_)
#     # 6)模型评估
#     y_predict = estimator.predict(x_test)
#     #print("predicted price: \n", y_predict)
#     error = mean_squared_error(y_test, y_predict)
#     score = estimator.score(x_test,y_test)
#     #print("正规方程-均方误差: \n", error)
#     return Bunch(estimator = model,score = score)
# def run_normalized(model,t):
#     linearresult = []
#     for i in range(0, 1000):
#         linear = linear_normalized(model,t)
#         dictresult = {"score": linear.score}
#         linearresult.append(dictresult)
#         # 保存模型
#         joblib.dump(model, dirname(__file__) +"/models_of_mice/"+str(model)[0:3]+"/"+"my_lr_" + str(linear.score) + ".pkl")
#     sorted_score = sorted(linearresult, key=operator.itemgetter('score'))
#     import re
#     import os
#     dirpath = join(dirname(__file__),"models_of_mice",str(model)[0:3])
#     for file in os.listdir(dirpath):
#         if not re.search(str(sorted_score[-1]["score"]),str(file)):
#             os.remove(dirpath+"/"+file)
#     print("maxscore of " + str(model) + ":\n", sorted_score[-1]["score"])
#     print("mdoel of " + str(model) + ":\n", sorted_score[-1])
#     maxdict[str(model)] = sorted_score[-1]["score"]
#     result2txt = "maxscore of " + str(model) + ":\n" + str(sorted_score[-1]["score"])
#     with open(scoretxt, 'a') as file_handle:
#         file_handle.write(result2txt)
#         file_handle.write('\n')
#
# if __name__ == "__main__":
#     # code true crispra_v1
#     crispra = load_crispra()
#     print("crispra: \n", crispra)
#     print("feature number: \n", crispra.data.shape)
#     mindict = {}
#     maxdict = {}
#     # for t in ():
#     t = 0.2
#     with open(scoretxt, 'a') as file_handle:
#         file_handle.write("\n")
#         file_handle.write("training set : " + str(100 * (1 - t)) + "%")
#         file_handle.write("\n")
#     run_normalized(ExtraTreeRegressor(), t)
#     run_normalized(XGBRegressor(), t)
#     run_normalized(RandomForestRegressor(), t)
#     run_normalized(AdaBoostRegressor(), t)
#     run_normalized(GradientBoostingRegressor(), t)
#     run_normalized(BaggingRegressor(), t)
#     # compare
#     sorted_max = max(maxdict, key=maxdict.get)
#     print("max score:\n" + sorted_max, maxdict[sorted_max])
#     result2txt = "max score:\n" + str(sorted_max) + str(maxdict[sorted_max])
#     with open(scoretxt, 'a') as file_handle:
#         file_handle.write('\n')
#         file_handle.write(result2txt)
#     # #predict with the trained model
#     # estimator = grabTree(dirname(__file__) + "/data/" + "my_lr_0.8944624178100227.pkl")
#     # crispra = load_crispra()
#     # x_train, x_test, y_train, y_test = train_test_split(crispra.data, crispra.target, test_size=0.2,random_state=11)
#     # print(x_train[0].reshape(1, -1))
#     # y_predict = estimator.predict(x_train[0].reshape(1, -1))
#     # print(y_predict)
#     # print(y_train[0].reshape(1, -1))
#     # print(y_predict/y_train[0].reshape(1, -1))
#
#
#
import numpy as np
from os.path import dirname
import joblib
import sys
# #predict with the trained model
user_train = sys.argv[1]
#print(str(user_train)+"fffff")
#user_train = "1.37401474,  0.33645973, 2.11035398,  1.8803562,   0.72672721 , 2.28216878 , 0.56584782,  1.82112032  ,0.88874096 , 0.72263392"
user_array = np.array(list(map(float,user_train.split(","))))
#print(user_array)
estimator_Bag =joblib.load(dirname(__file__) + "/models_of_mice/" + "Bag/" +"my_lr_0.839061447484988.pkl")
estimator_Ext =joblib.load(dirname(__file__) + "/models_of_mice/" + "Ext/" +"my_lr_0.8325484983483677.pkl")
estimator_Ran =joblib.load(dirname(__file__) + "/models_of_mice/" + "Ran/"+"my_lr_0.8388369453808923.pkl")
estimator_XGB =joblib.load(dirname(__file__) + "/models_of_mice/" + "XGB/"+"my_lr_0.8477100097550855.pkl")
#estimator_Ada =joblib.load("/Users/cuijiajun/Desktop/SRTP/python/ml_crispra_v4" + "/models_of_mice/" + "Ada/"+"my_lr_0.8368928911176641.pkl")
#estimator_Gra =joblib.load("/Users/cuijiajun/Desktop/SRTP/python/ml_crispra_v4" + "/models_of_mice/" + "/Gra/"+"my_lr_0.8496595129178548.pkl")

y_predict_Ran = estimator_Ran.predict(user_array.reshape(1,-1))[0]
print(str(y_predict_Ran))
y_predict_XGB = estimator_XGB.predict(user_array.reshape(1,-1))[0]
print(str(y_predict_XGB))
y_predict_Bag = estimator_Bag.predict(user_array.reshape(1,-1))[0]
print(str(y_predict_Bag))
y_predict_Ext = estimator_Ext.predict(user_array.reshape(1,-1))[0]
print(str(y_predict_Ext))
#y_predict_Ada = estimator_Ada.predict(user_array.reshape(1,-1))[0]
#print(str(y_predict_Ada))
#y_predict_Gra = estimator_Gra.predict(user_array.reshape(1,-1))[0]
#print(str(y_predict_Gra))
y_predict = [y_predict_Ran,y_predict_XGB,y_predict_Bag,y_predict_Ext]
print(sum(y_predict)/len(y_predict))
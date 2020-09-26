import csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from os.path import join, dirname
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.utils import Bunch
from sklearn.preprocessing import MinMaxScaler
import operator
scoretxt = 'max_scores_v4_log10_7feautures.txt'
#scoretxt = 'dt_parameter.txt'
#scoretxt = 'gc_abnormaldone_max_scores.txt'
def load_crispra(*, return_X_y=False):
    module_path = dirname(__file__)

    #data_file_name = join(module_path, 'data', 'crispra_v3_gc.csv')
    #scores all positive
    data_file_name = join(module_path, 'data', 'crispra_v3_gc_allpositive_0-0.01_Rstudio_log10_features_7.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp =next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,))
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.float64)
        # #data normalization
        # minmaxtransfer = MinMaxScaler()
        # data = minmaxtransfer.fit_transform(data)
    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 filename=data_file_name)
def grabTree(filename):
    fr = open(filename, 'rb')
    return joblib.load(fr)
def linear_normalized(model,test_size):
    # 1） 获取数据
    crispra = load_crispra()
    #print("crispra: \n", crispra)
    #print("feature number: \n",crispra.data.shape)
    x_train, x_test, y_train, y_test = train_test_split(crispra.data, crispra.target, test_size=test_size)

    # 2)
    # 3) 标准化z-score
    print(type(x_train))
    print("then")
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    print(type(x_train))
    # # data normalization
    # minmaxtransfer = MinMaxScaler()
    # x_train = minmaxtransfer.fit_transform(x_train)
    # x_test = minmaxtransfer.fit_transform(x_test)
    # 加载模型
    #estimator = joblib.load("my_lr.pkl")
    #estimator = grabTree(dirname(__file__) + "/data/" + "my_lr_0.8944624178100227.pkl")

    # # 4) 预估器
    estimator=model
    estimator.fit(x_train,y_train)
    # 5)得出model
    #print("正规方程权重系数为：\n",estimator.coef_)
    #print("正规方程偏置为：\n", estimator.intercept_)
    # 6)模型评估
    y_predict = estimator.predict(x_test)
    #print("predicted price: \n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    score = estimator.score(x_test,y_test)
    # 保存模型
    #joblib.dump(estimator, "my_lr.pkl")
    #print("正规方程-均方误差: \n", error)
    if str(model) in ["LinearRegression()","SGDRegressor()","Ridge()"]:
        return Bunch(estimator = model,score = score,error=error, coef=estimator.coef_, intercept=estimator.intercept_)
    else:
        return Bunch(estimator = model,score = score)
def run_normalized(model,t):
    linearresult = []
    if str(model) in ["LinearRegression()", "SGDRegressor()", "Ridge()"]:
        for i in range(0, 1000):
            linear = linear_normalized(model,t)
            dictresult = {"score": linear.score, "error": linear.error, "coef": linear.coef,
                          "intercept": linear.intercept}
            linearresult.append(dictresult)
        sorted_error = sorted(linearresult, key=operator.itemgetter('error'))
        sorted_score = sorted(linearresult, key=operator.itemgetter('score'))

        print("minerror of "+str(model)+":\n", sorted_error[0]["error"])
        print("mdoel1 of "+str(model)+":\n", sorted_error[0])
        print("maxscore of "+str(model)+":\n", sorted_score[-1]["score"])
        print("mdoel2 of "+str(model)+":\n", sorted_score[-1])
        mindict[str(model)] = sorted_error[0]["error"]
        maxdict[str(model)] = sorted_score[-1]["score"]
        result2txt = "maxscore of " + str(model) + ":\n" + str(sorted_score[-1]["score"])
        with open(scoretxt, 'a') as file_handle:
            file_handle.write(result2txt)
            file_handle.write('\n')
    else:
        for i in range(0, 1000):
            linear = linear_normalized(model,t)
            dictresult = {"score": linear.score}
            linearresult.append(dictresult)
        sorted_score = sorted(linearresult, key=operator.itemgetter('score'))
        print("maxscore of " + str(model) + ":\n", sorted_score[-1]["score"])
        print("mdoel2 of " + str(model) + ":\n", sorted_score[-1])
        maxdict[str(model)] = sorted_score[-1]["score"]
        result2txt = "maxscore of " + str(model) + ":\n" + str(sorted_score[-1]["score"])
        with open(scoretxt, 'a') as file_handle:
            file_handle.write(result2txt)
            file_handle.write('\n')

if __name__ == "__main__":
    # code true crispra_v1
    crispra = load_crispra()
    print("crispra: \n", crispra)
    print("feature number: \n", crispra.data.shape)
    mindict = {}
    maxdict = {}
    # for t in ():
    t = 0.2
    with open(scoretxt, 'a') as file_handle:
        file_handle.write("\n")
        file_handle.write("training set : " + str(100 * (1 - t)) + "%")
        file_handle.write("\n")
    #代码1 正规方程的优化方法
    run_normalized(LinearRegression(),t)
    #代码2 梯度下降的优化方法
    # run_normalized(SGDRegressor(),t)
    # # 代码3 岭回归的优化方法
    # run_normalized(Ridge(),t)
    # #other models_of_human
    # run_normalized(SVR(),t)
    # run_normalized(Lasso(),t)
    # #跑不下来（10min）done
    # #run_normalized(MLPRegressor(max_iter=5000),t)
    # run_normalized(DecisionTreeRegressor(),t)
    # run_normalized(AdaBoostRegressor(),t)
    # run_normalized(ExtraTreeRegressor(),t)
    # run_normalized(XGBRegressor(),t)
    # run_normalized(RandomForestRegressor(),t)
    # run_normalized(KNeighborsRegressor(), t)
    # run_normalized(GradientBoostingRegressor(),t)
    # run_normalized(BaggingRegressor(),t)
    # #compare
    # ##sorted_min = min(mindict,key=mindict.get)
    # ##print("min error:\n"+sorted_min,mindict[sorted_min])
    # sorted_max = max(maxdict, key=maxdict.get)
    # print("max score:\n"+sorted_max, maxdict[sorted_max])
    # result2txt = "max score:\n"+str(sorted_max) + str(maxdict[sorted_max])
    # with open(scoretxt, 'a') as file_handle:
    #     file_handle.write('\n')
    #     file_handle.write(result2txt)
    # # #optimize the extratreeregressor and the decisiontreeregressor
    # for x in range(10,100):
    # #for n in range(10,100):
    #     #run_normalized(BaggingRegressor(base_estimator=DecisionTreeRegressor(max_leaf_nodes=36),n_estimators=x),t)
    #     run_normalized(DecisionTreeRegressor(max_leaf_nodes =x),t)
    # run_normalized(DecisionTreeRegressor(), t)
    # #compare
    # sorted_max = max(maxdict, key=maxdict.get)
    # print("max score:\n"+sorted_max, maxdict[sorted_max],"nodes:",x)
    # with open(scoretxt, 'a') as file_handle:
    #     file_handle.write('\n')
    #     file_handle.write("max score: "+str(sorted_max) + "\n" + str(maxdict[sorted_max]))
    # #27 best (log10)
    # # # 最大叶子节点数max_leaf_nodes: 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。
    # #meishakeyitiaode prameter only (max_leaf_nodes)
    # #dt nodes=17 get max score
    # #run_normalized(DecisionTreeRegressor(max_leaf_nodes=17),t)
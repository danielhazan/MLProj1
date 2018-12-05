#import numpy as np
from numpy import *
import pandas as pd

import matplotlib.pyplot as plt



data = pd.read_csv("kc_house_data.csv")

data = data.drop(["date","id","sqft_living","long","lat", "condition","yr_renovated","sqft_living15","sqft_lot15"],axis=1)

data= data.dropna(axis= 0,how= "any")

data = pd.get_dummies(data,columns=["zipcode"])
#removing all data with illogical data, i.e negative price, negative bedrooms etc.
data.drop(data[data["price"]<=0].index,inplace=True)
data.drop(data[data["bathrooms"]<=0].index,inplace=True)
data.drop(data[data["floors"]<0].index,inplace=True)
data.drop(data[data["bedrooms"]<=0].index,inplace= True)


data.insert(loc=0,column= "intercept",value=[1 for k in range(len(data))])

"""this function seperate the data to training data and test data
@:param p - > factor with which to devide the data"""
def seperate_data(p):
    rs = random.rand(len(data)) < p/100
    training_data = data[rs]
    testing_data = data[~rs]
    return training_data, testing_data

"""this function calculate the ERM i.e the loss(error) function of
 the target_set and the predicted data given by the formula Ls =(1/m)*(<w,x>-y)**2
 where <w,x> is the predicted data found by the regression(normal equations) and y
 is the target set,the real data, and m is the number of examples
 @:param real_y -> target set i.e training price
 @:param predicted_y -> prediction of prices taken from normal equations"""
def least_squares(real_y,predicted_y):
    return (linalg.norm(predicted_y-real_y)**2)/len(real_y)

def linear_regression():
    test_error, train_error = list(), list()

    for k in range(1,100):
        training_data, testing_data = seperate_data(k)

        train_pr, test_pr = training_data["price"], testing_data["price"]

        training_data = training_data.drop(["price"], axis=1)
        testing_data = testing_data.drop(["price"],axis=1)

        #calculating pseudo inverse
        pinv = linalg.pinv(training_data)

        #calculating the w vector (solution of the norm equations)
        w= dot(pinv, train_pr)

        traininig_error = least_squares(train_pr,dot(training_data,w))
        testing_error = least_squares(test_pr,dot(testing_data,w))

        train_error.append(traininig_error)
        test_error.append(testing_error)

    plt.plot([k for k in range(1,100)],train_error,label= "trainErr(ESM)")
    plt.plot([k for k in range(1,100)],test_error,label= "testErr(ESM)")
    plt.xlabel("the percent of trained data ")
    plt.ylabel("ERM")
    plt.title("ERM as function of ammount of training data")
    plt.legend()
    plt.show()


linear_regression()

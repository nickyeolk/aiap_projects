# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 08:52:31 2018
Numpy implementation of nn
@author: likkhian
"""
import numpy as np


class MLPTwoLayers:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # print('nihao', self.input_size, self.hidden_size, self.output_size)
        # initialize weights
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        self.b1 = np.ones(self.hidden_size)
        self.b2 = np.ones(self.output_size)

    def __sigmoid(self, x):
        # exps = np.exp(x - np.max(x))
        return 1 / (1 + np.exp(-x))

    def __sigmoid_prime(self, z):
        return self.__sigmoid(z)*(1-self.__sigmoid(z))

    def __softmax(self, smx):
        smxx = smx-np.max(smx)
        return np.exp(smxx)/np.sum(np.exp(smxx))

    def forward(self, X):
        self.X = X
        # print('shape check', np.shape(self.X), np.shape(self.w1),
        # np.shape(self.b1), np.shape(self.w2), np.shape(self.b2))
        self.x1 = np.dot(self.X.T, self.w1) + self.b1
        self.x1a = self.__sigmoid(self.x1)
        # print('x1', np.shape(self.x1), np.shape(self.x1a))
        self.x2 = np.dot(self.x1a, self.w2) + self.b2
        self.x2a = self.__softmax(self.x2)
        # print('x2', np.shape(self.x2), np.shape(self.x2a))
        return self.x2a

    def loss(self, y_pred, y):
        self.y_pred = y_pred
        # print(self.y_pred)
        self.y = y
        # print(self.y)
        # self.y_pred[self.y] = self.y_pred[self.y]-1
        return -np.log(self.y_pred[self.y])

    def backward(self, loss):
        self.lr = 1e-3
        self.lossx = self.y_pred-self.y
        self.t1_2 = self.lossx
        # self.t2_2 = self.__sigmoid_prime(self.x2)
        self.t3_2 = self.x1a
        # print(np.shape(self.t1_2),np.shape(self.t2_2), np.shape(self.t3_2),np.shape(self.w2))
        self.big_delt2 = np.dot(self.t3_2.reshape(-1,1),self.t1_2.reshape(1,-1))# *self.t2_2.reshape(1,-1)) 
        self.t1_1 = np.dot(self.w2,self.t1_2*self.__sigmoid_prime(self.x2))
        self.t2_1 = self.__sigmoid_prime(self.x1)
        self.t3_1 = self.X
        self.big_delt1 = np.dot(self.t3_1.reshape(-1, 1),self.t1_1*self.t2_1.reshape(1, -1))

        # print(self.w2[:5,:5])
        self.w2 = self.w2 - self.big_delt2*self.lr
        self.w1 = self.w1 - self.big_delt1*self.lr
        self.b2 = self.b2 - self.t1_2*self.lr
        self.b1 = self.b1 - self.t1_1*self.t2_1*self.lr
        
        # self.bigdelt1 = 0
        # self.bigdelt2 = 0
        # self.delta2 = self.lossx*self.__sigmoid_prime(self.x2)
        # self.delta1 = np.dot(self.w2, self.delta2)*self.__sigmoid_prime(self.x1)
        # self.bigdelt2 = np.dot(self.delta2, self.x2a) #partial derivative of total error w.r.t. the weight
        # self.bigdelt1 = np.dot(self.delta1, self.x1a)
        # print('shapes delta2,w1,w2,x2a,x1a,',np.shape(self.delta2),np.shape(self.w1),
        #       np.shape(self.w2),np.shape(self.x2a),np.shape(self.x1a))
        # print('shapes bigdelt',(self.bigdelt1),(self.bigdelt2))
        # self.w2 = self.w2-self.bigdelt2*self.lr
        # self.w1 = self.w1-self.bigdelt1*self.lr
        # self.b2 = self.b2
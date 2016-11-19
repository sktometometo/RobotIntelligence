#:coding=utf-8:

import numpy as np
import random
import struct
import sys
import time
import pickle

def PutBar(per, barlen):
    perb = int(per/(100.0/barlen))

    s = '\r'
    s += '|'
    s += '#' * perb
    s += '-' * (barlen - perb)
    s += '|'
    s += ' ' + (str(per) + '%').rjust(4)

    sys.stdout.write(s)

def sigmoid(a,z):
    return 1/(1+np.exp(-a*z))

def sigmoid_d(a,z):
    return a*sigmoid(a,z)*(1-sigmoid(a,z))

class nn:
    def __init__(self,lst_unitnum,a,learning_rate):
        self.a = a
        self.learning_rate = learning_rate
        self.layer_num = len(lst_unitnum)
        self.weightslst = []
        for i in range(0,self.layer_num-1):
            self.weightslst.append(np.ones((lst_unitnum[i+1],lst_unitnum[i]+1)))

    def save(self,fname):
        with open(fname, mode='wb') as f:
            pickle.dump(self,f)

    def loat(self,fname):
        with open(fname, mode='wb') as f:
            lobj = pickle.load(f)
        self.a = lobj.a
        self.learning_rate = lobj.learning_rate
        self.layer_num = lobj.layer_num
        self.weightslst = lobj.weightslst

    def fit(self,X,Y):
        test_num = len(Y)
        Indlst = np.random.randint(0,test_num-1,test_num-1)
        hoge = 0
        for i in Indlst:
            hoge += 1
            for j in range(0,100):
                outputs = []
                outputs.append(X[i]) # 入力層の出力
                for k in range(0,self.layer_num-1):
                    outputs.append(sigmoid(self.a,np.dot(self.weightslst[k],np.append(outputs[-1],1))))
                    # 各層の出力を計算してい

                # 誤差逆伝播法
                if np.linalg.norm(Y[i]-outputs[-1]) < 0.00001:
                    break
                #delta = (Y[i]-outputs[-1])*self.a*outputs[-1]*(np.ones(len(outputs[-1]))-outputs[-1])
                #print(delta.size)
                self.weightslst[-1] = self.weightslst[-1] - self.learning_rate * np.array(np.transpose(np.mat(Y[i]-outputs[-1])) * np.mat(np.append(outputs[-2],1)))
                for k in range(1,self.layer_num-2):
                    delta = np.dot(self.weightslst[-1-k],delta) * (self.a*outputs[-1-k]*(np.ones(len(outputs[-1-k]))-outputs[-1-k]))
                    self.weightslst[-1-k] = self.weightslst[-1-k] - self.learning_rate * np.transpose(np.mat(delta)) * np.mat(np.append(outputs[-1-k],1))
            PutBar(hoge*100/test_num, 30)

    def predict_one(self,xi):
        output = xi
        for i in range(0,self.layer_num-1):
            output = sigmoid(self.a,np.dot(self.weightslst[i],np.append(output,1)))
        return output

    def predict(self,X):
        outputs = []
        ItemNum = len(X)
        for i in range(0,ItemNum):
            outputs.append(self.predict_one(X[i]))
        return outputs

def Xread(fname,r):
    f_x = open(fname,'rb')
    X = []
    f_x.read(4);
    ItemNum = struct.unpack('>i',f_x.read(4))[0];
    Num_row = struct.unpack('>i',f_x.read(4))[0];
    Num_col = struct.unpack('>i',f_x.read(4))[0];
    buf = np.ones((Num_row*Num_col))
    for i in range(0,ItemNum):
        for j in range(0,Num_row*Num_col):
            buf[j] = struct.unpack('B',f_x.read(1))[0]*1.0/r
        X.append(buf)
    return X

def Yread(fname):
    f_y = open(fname,'rb')
    Y = []
    f_y.read(4)
    ItemNum = struct.unpack('>i',f_y.read(4))[0]
    for i in range(0,ItemNum):
        buf = np.zeros((10))
        buf[struct.unpack('B',f_y.read(1))[0]] = 1
        Y.append(buf)
    return Y

if __name__ == "__main__":
    # Todo 入力ベクトルの正規化処理の追加
    train_x = Xread('MNIST/train-images-idx3-ubyte',255)
    train_y = Yread('MNIST/train-labels-idx1-ubyte')
    test_x = Xread('MNIST/t10k-images-idx3-ubyte',255)
    test_y = Yread('MNIST/t10k-labels-idx1-ubyte')

    lst_unitnum = [28*28,100,10]
    a = 10
    learning_rate = 0.1

    neuralnet = nn(lst_unitnum,a,learning_rate)
    print("start learning...")
    neuralnet.fit(train_x,train_y)
    print("learning completed.")
    neuralnet.save('nn.data')

    test_outputs = neuralnet.predict(test_x)
    ItemNum = len(test_outputs)
    correctnum = 0
    for i in range(0,ItemNum):
        if np.linalg.norm(test_outputs[i]-test_y[i]) < 0.0001:
            correctnum = correctnum + 1
    print("accuracy:"+str(correctnum*1.0/ItemNum))

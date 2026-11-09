# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:14:31 2022

@author: Moteki
"""
import numpy as np
fname= 'simulated_XY_dataset.csv' # CloudChamberFowardCalc.pyの出力ファイル
dataset = np.loadtxt(fname, delimiter=',', skiprows=1) #カンマ区切りテキストファイルをNumpy配列 (ndarray)として読み込み
X= dataset.transpose()[0:2,:] # 式(124)の行列X 
Y= dataset.transpose()[2:4,:]
Y= np.insert(Y,0,1,axis=0) # 式(124)の行列Y (1だけからなる行を1行目に挿入)
A= 　 　　　　#右辺を埋めてください。 式(127)の計算式をPythonで書いたものです。
print(A) #行列をAをコンソール画面に表示

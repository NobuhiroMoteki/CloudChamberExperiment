# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:59:46 2022

@author: Moteki
"""
import numpy as np
fname= 'simulated_XY_dataset.csv'
dataset = np.loadtxt(fname, delimiter=',', skiprows=1)
X= dataset.transpose()[0:2,:]
Y= dataset.transpose()[2:4,:]
Y= np.insert(Y,0,1,axis=0)
A= np.matmul(np.matmul(X,Y.transpose()),np.linalg.inv(np.matmul(Y,Y.transpose())))
print(A)

import csv
import atmospheric_composition
dry_air= atmospheric_composition.DryAir()
water= atmospheric_composition.Water()

#========== parameters for calcualting the saturation vapor pressure =============
a_param= 2.5368e11 # The constant "a" [Pa] determined in the Problem 2.2
b_param= 5.4205e3 #The constant "b" [K] determined in the Problem 2.2
#=================================================================================


fname_cloudchamberdata='cloud_chamber_data_2022b_center.csv'

with open('processed_data2022.csv','w') as fout:
    labels= 'Time_stamp,qw_gkg_ref,nd_ref,qw_gkg_pred,nd_pred'
    fout.write(labels + '\n')
    with open(fname_cloudchamberdata,'r') as fin:
        dataReader= csv.reader(fin)
        for row in dataReader:
            if 'Time_stamp' in row : continue
            val_list= [float(valstr) for valstr in row]
            time_stamp= val_list[0]
            T_ini= val_list[1] # [K]
            p_ini= val_list[2]*1e2 # [Pa]
            zeta_moist= val_list[6]
            zeta_turbid= val_list[7]
            ref_T= val_list[8]
            ref_RH= val_list[9]
            ref_particleconc= val_list[10]
            
            #===== 市販センサの相対湿度 ref_RH [%] を水蒸気混合比 qw_ref [g/(kg dry air)]に変換 =============
            e_Pa= ref_RH/100*water.es(ref_T+273.15,a_param,b_param)
            qw_ref= water.eps*e_Pa/p_ini
            qw_ref= qw_ref*1e3 # [g/(kg dry air)]
            #==========================================================================================
            
            #=== 市販センサのエアロゾル数濃度[#/cm3] ref_particleconc を 数混合比 nd_ref [#/(kg dry air)]に変換 ====
            rho= p_ini/(dry_air.Rd*T_ini) # [kg/m3] density of dry air 
            nd_ref= ref_particleconc*1e6/rho
            #===============================================================================================
            
            #===== 回帰モデルによる逆作用素の近似式(ユーザーが指定)=========
            y_tilde= np.array([1,zeta_moist,zeta_turbid])
            x= np.dot(A,y_tilde)
            qw_pred= x[0]
            log10nd_pred= x[1]
            #========================================================
          
            nd_pred= 10**log10nd_pred
            
            datastr= "{:.0f},{:5g},{:5g},{:5g},{:5g}".format(time_stamp,qw_ref,nd_ref,qw_pred,nd_pred)
            print(datastr)
            fout.write(datastr+'\n')
            
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:59:46 2022

@author: Moteki
"""
import numpy as np
import csv
import atmospheric_composition
dry_air= atmospheric_composition.DryAir()
water= atmospheric_composition.Water()

#=========================================================================================
#ComputeMatrixA.py のコードの必要部分を貼り付けて、ここでsimulated_XY_dataset.csvを読み込んで行列Aを算出してください


#=========================================================================================

#========== parameters for calcualting the saturation vapor pressure =============
a_param=  # The constant "a" [Pa] determined in the Problem 2.2
b_param=  #The constant "b" [K] determined in the Problem 2.2
#=================================================================================


fname_cloudchamberdata=    # 観測データファイルの名前を文字列で指定

with open('processed_data.csv','w') as fout:
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
    
            qw_ref=  
            #==========================================================================================
            
            
            #=== 市販センサのエアロゾル数濃度[#/cm3] ref_particleconc を 数混合比 nd_ref [#/(kg dry air)]に変換 ====
            
            nd_ref= 
            #===============================================================================================
            
            
            #===== 式(123)に従い、行列Aを使って(zeta_moist, zeta_turbid) から (qw,log10nd)を算出　====================
           
            qw_pred=   # [g/(kg dry air)]
            log10nd_pred=  
            
            nd_pred= 10**log10nd_pred # [#/(kg dry air)]
            #=================================================================================================
          
            
            datastr= "{:.0f},{:5g},{:5g},{:5g},{:5g}".format(time_stamp,qw_ref,nd_ref,qw_pred,nd_pred)
            print(datastr)
            fout.write(datastr+'\n') # データをファイルへ書き出し
            
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:53:03 2022

@author: Moteki
"""

import pandas as pd
import numpy as np
import atmospheric_composition
dry_air= atmospheric_composition.DryAir()
water= atmospheric_composition.Water()

#========== parameters for calcualting the saturation vapor pressure =============
a_param= 2.56e11 # The constant "a" [Pa] determined in the Problem 2.2
b_param= 5.42e3 #The constant "b" [K] determined in the Problem 2.2
#=================================================================================


#fname= 'simulated_time_profile.csv'
fname= 'cloud_chamber_data_2022b_left.csv'
#fname= 'processed_data.csv'

df= pd.read_csv(fname)
df=df.astype('float')
print(df,'\n')

print(df['T_ini'].describe(),'\n')
print(df['p_ini'].describe(),'\n')
print(df['p_offset'].describe(),'\n')
print(df['tau_vac'].describe(),'\n')
print(df['ref_T'].describe(),'\n')
print(df['ref_RH'].describe(),'\n')
print(df['ref_particleconc'].describe(),'\n')

#===== 飽和比から水蒸気混合比 [g/(kg dry air)]に変換 =======
es_Pa= a_param*np.exp(-b_param/(df['ref_T']+273.15))
e_Pa= df['ref_RH']/100*es_Pa
qw_ref= water.eps*e_Pa/(df['p_ini']*1e2)
qw_ref= qw_ref*1e3 # [g/(kg dry air)]
df['ref_qw']= qw_ref
#========================================================
print(df['ref_qw'].describe(),'\n')

#===== 数濃度[#/cm3]から数混合比 [#/(kg dry air)]に変換 ====
rho= (df['p_ini']*1e2)/(dry_air.Rd*(df['ref_T']+273.15)) # [kg/m3] density of dry air 
nd_ref= df['ref_particleconc']*1e6/rho
df['ref_nd']= nd_ref
#========================================================
print(df['ref_nd'].describe(),'\n')
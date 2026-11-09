# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:53:03 2022

@author: Moteki
"""

import pandas as pd
from datetime import datetime, timedelta

#fname= 'simulated_time_profile.csv'
#fname= 'cloud_chamber_data_2022b_center.csv'
fname= 'processed_data2022.csv'

df= pd.read_csv(fname)
df=df.astype('float')
print(df,'\n')

#============= Time_stampから年月日日時へ変換 ==========================
base_date = datetime(1970,1,1) # January 1st, 1970 at midnight
delta_date_list = [timedelta(seconds= v)  for v in df['Time_stamp']]
df['Date'] = [base_date + v for v in delta_date_list]
#====================================================================

print(df)
print(df['nd_ref'])

# simple x, y plot
# df.plot(x='Date', y='ref_RH')
# df.plot(x='time_s', y='Tr')

# append another plot
ax = df.plot(x='Date', y='qw_gkg_ref')
df.plot(x='Date', y='qw_gkg_pred', ax=ax, ylim=[0,20])

ax = df.plot(x='Date', y='nd_ref')
df.plot(x='Date', y='nd_pred', ax=ax, logy=True, ylim=[1e6,1e11])

sz=1
ax= df.plot.scatter(x='qw_gkg_ref', y='qw_gkg_pred', s=sz, marker='.', grid=True, xlim=[5,20], ylim=[5,20])
#ax.set_aspect('equal')


ax= df.plot.scatter(x='nd_ref', y='nd_pred', s=sz, marker='.', grid=True, xlim=[1e6,1e8], ylim=[1e9,1e10])
#ax.set_aspect('equal')



# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:53:03 2022

@author: Moteki
"""

import pandas as pd
from datetime import datetime, timedelta



fname= 'simulated_time_profile.csv'
#fname= 'cloud_chamber_data_2022b_center.csv'
#fname= 'processed_data.csv'

df= pd.read_csv(fname)
df=df.astype('float')
print(df,'\n')

#============= Time_stampから年月日日時へ変換 ==========================
# base_date = datetime(1970,1,1) # January 1st, 1970 at midnight
# delta_date_list = [timedelta(seconds= v)  for v in df['Time_stamp']]
# df['Date'] = [base_date + v for v in delta_date_list]
#=====================================================================

print(df)

# y0の時系列プロット
df.plot(x='time_s', y='S')

# y1,y2の時系列プロットの重ね書き
# ax = df.plot(x='Date', y='y1')
# df.plot(x='Date', y='y2', ax=ax)






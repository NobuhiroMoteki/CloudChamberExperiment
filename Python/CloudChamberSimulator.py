# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:21:51 2022

@author: Moteki
"""

import math
import chamber
import atmospheric_composition
import miescat

a_param=  # The constant "a" [Pa] determined in the Problem 2.2
b_param=  #The constant "b" [K] determined in the Problem 2.2

T_ini=  # [K] 初期温度
p_ini=  # [Pa] 初期圧力
tau_vac=  # [s] 減圧パラメータ tau_vac
p_offset=  # [Pa] 減圧パラメータ p_offset

qw=  # total water mixing ratio [kg/(kg dry air)]
nd=  # number concentration of aerosol [#/(kg dry air)]
a_ini= 50e-9 # [m] aerosol particle radius (assumed)

dry_air= atmospheric_composition.DryAir()
water= atmospheric_composition.Water()
cloud_chamber= chamber.CloudChamber(T_ini= T_ini, p_ini= p_ini, tau_vac= tau_vac, p_offset= p_offset, L= 0.2)

T= T_ini
a= a_ini

time_s= 0
dtime= 1e-3

with open('simulated_time_profile.csv','w') as fout:
    labels= 'time_s,P_Pa,T_K,e_Pa,qv,S,a_um,Qext,Tr'
    fout.write(labels + '\n')
    while time_s <= 4:
        
        # --- current pressure and its time derivative ---
        P= cloud_chamber.P(time_s)
        dP_dt= cloud_chamber.dP_dt(time_s)
        # ------------------------------------------------
        
        # --- compute current transmittance Tr ---
        n= nd*P/(dry_air.Rd*T) # number concentration of droplet [#/m3]
        Qsca,Qext,Qabs,Qback,S1,S2= miescat.gMIE_NonAbsMed(wl_0= 670e-9,r_p= a,eper_m= dry_air.eper,mper_m= dry_air.mper,eper_p= water.eper, mper_p= water.eper)
        Cext= math.pi*a**2*Qext # extinction cross section of each droplet [m2]
        Tr= math.exp(-Cext*n*cloud_chamber.L)
        #-----------------------------------------
        
        ql= 4*math.pi/3*water.densl*a**3*nd # liquid water mixing ratio [kg/ (kg dry air)]
        qv= qw-ql # water vapor mixing ratio [kg/ (kg dry air)]
        e= qv*P/water.eps # vapor pressure [Pa]
        S= e/water.es(T,a_param,b_param)
        
        #--- compute droplet growth rate---
        if S>=1 :
            #-- compute droplet temperature T0 by Newton method
            AA= water.lv*water.Dv/water.Rv
            BB= S*water.es(T,a_param,b_param)/T
            CC= dry_air.ka*T
            x= T
            while True:
                func= -AA*water.es(x,a_param,b_param)/x-dry_air.ka*x+AA*BB+CC
                d_func= -AA*a_param/x**2*(b_param/x-1)*math.exp(-b_param/x)-dry_air.ka
                x1= x-func/d_func
                if abs(x1-x) < 1e-3:
                    break
                x= x1
            T0= x
            da_dt= (1/a)*(S*water.es(T,a_param,b_param)/T-water.es(T0,a_param,b_param)/T0)*(water.Dv/(water.densl*water.Rv)) 
        else:
            da_dt= 0
        #----------------------------------
        
        dqv_dt= -4*math.pi*water.densl*nd*a**2*da_dt
        cp= dry_air.cp+qv*water.cpv+ql*water.cl # specific heat of moist air 
        dT_dt= ((water.Rv*T-water.lv)*dqv_dt+T*(water.Rv*qv+dry_air.Rd)*dP_dt/P)/cp
        
        T= T+dtime*dT_dt
        a= a+dtime*da_dt
        
        print(time_s,a*1e6,T,S)
        datastr= "{:5g},{:5g},{:5g},{:5g},{:5g},{:5g},{:5g},{:5g},{:5g}".format(time_s,P,T,e,qv,S,a*1e6,Qext,Tr)
        fout.write(datastr + '\n')
        
        time_s += dtime
    
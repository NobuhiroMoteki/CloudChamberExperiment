# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:21:51 2022

@author: Moteki
"""

import math
import numpy as np
import chamber
import atmospheric_composition
import miescat

#========== parameters for calcualting the saturation vapor pressure =============
a_param= 2.5368e11 # The constant "a" [Pa] determined in the Problem 2.2
b_param= 5.4205e3 #The constant "b" [K] determined in the Problem 2.2
#=================================================================================

#========== representative value of each parameter of the operating cloud chamber ==============
T_ini_repr= 299.86
p_ini_repr= 1002e2
tau_vac_repr= 3.9474
p_offset_repr= 441.34e2
cloud_chamber= chamber.CloudChamber(T_ini= T_ini_repr, p_ini= p_ini_repr, tau_vac= tau_vac_repr, p_offset= p_offset_repr, L= 0.2)
#===============================================================================================

dry_air= atmospheric_composition.DryAir()
water= atmospheric_composition.Water()

a_ini= 50e-9 # [m] aerosol particle radius

#=========== input parameter range for simulated data ================
# total water mixing ratio [g/(kg dry air)]
qw_min= 7
qw_max= 16
numgrid_qw= 10
qw_gird= np.linspace

# log10 of {number of cloud condensation nuclei (aerosol) per unit mass of air [#/kg]}
log10nd_min= 9
log10nd_max= 11
numgrid_log10nd= 10
#====================================================================

#=========== feature parameters ================
Tr_threshold= 0.985
dtime_eval= 0.3 # [s]
#================================================



with open('simulated_XY_dataset.csv','w') as fout:
    labels= 'qw,log10nd,zeta_moist,zeta_turbid'
    fout.write(labels + '\n')
    for qw in np.linspace(qw_min,qw_max,numgrid_qw):
        qw= qw*1e-3 # convert the unit to [kg/(kg dry air)]
        for log10nd in np.linspace(log10nd_min,log10nd_max,numgrid_log10nd):
            nd= 10**log10nd
            T= T_ini_repr
            a= a_ini
            
            time_s= 0
            dtime= 1e-3
            
            zeta_moist_determined= False
            zeta_turbid_determined= False
            zeta_moist= 0
            zeta_turbid= 0
    
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
                
                if (not zeta_moist_determined) and Tr <= Tr_threshold:
                    time_pc= time_s
                    zeta_moist= 1-P/p_ini_repr
                    zeta_moist_determined= True
                if (not zeta_turbid_determined) and zeta_moist_determined and time_s >= time_pc+dtime_eval:
                    zeta_turbid= (Tr_threshold-Tr)/dtime_eval
                    zeta_turbid_determined= True
                    
                
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
                time_s += dtime
                
            datastr= "{:5g},{:5g},{:5g},{:5g}".format(qw*1e3,math.log10(nd),zeta_moist,zeta_turbid)
            print(datastr)
            fout.write(datastr + '\n')
    
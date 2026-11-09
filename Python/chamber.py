# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:10:02 2022

@author: Moteki
"""
class CloudChamber:
    def __init__(self, T_ini= 297.15, p_ini= 1000e2, tau_vac= 3.7, p_offset= 450e2, L= 0.2):
        self.T_ini= T_ini
        self.p_ini= p_ini
        self.tau_vac= tau_vac
        self.p_offset= p_offset
        self.L= L
    
    def P(self, time_s):
        """
        Parameters
        ----------
        t : TYPE real
            elapsed time[s] from the start of evacuation

        Returns
        -------
        Pressure [Pa]

        """
        import math
        val= (self.p_ini-self.p_offset)*math.exp(-time_s/self.tau_vac)+self.p_offset
        return val
    
    def dP_dt(self, time_s):
        """
        Parameters
        ----------
        t : TYPE real
            elapsed time[s] from the start of evacuation

        Returns
        -------
        time derivative of Pressure [Pa s-1]

        """
        import math
        val= -(self.p_ini-self.p_offset)/self.tau_vac*math.exp(-time_s/self.tau_vac)
        return val
        

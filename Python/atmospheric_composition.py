# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:57:26 2022

@author: Moteki
"""
class DryAir:
    # dry air
    def __init__(self):
        self.Rd= 287.04 # [J kg-1 K-1]  gas constant
        self.cp= 1005 # [J kg-1 K-1] specific heat at constant P
        self.ka= 0.025 # [W m-1 K-1] thermal conductivity of dry air @ 0~20C
        self.eper= 1+0j # permittibity of air
        self.mper= 1+0j # permiability of air

class Water:
    # water
    def __init__(self):
        self.eps= 0.622 # ratio of molecular weights of water vapor to dry air
        self.Rv= 461.5 # [J kg-1 K-1]  gas constant
        self.cpv= 1850 # [J kg-1 K-1] specific heat of vapor at constant P
        self.Dv= 2.5e-5 # [m2 s-1] diffusion coefficient of water vapor in air @ 0~20C
        self.cl= 4128 # [J kg-1 K-1] specific heat of liquid water
        self.densl= 1.0e3 # [kg m-3] liquid water density
        self.eper= (1.33+0j)**2 # permittivity of liquid water @ visible wavelength
        self.mper= 1+0j # permiability of liquid water @ visible wavelength 
        self.lv= 2.501e6 # [J kg-1] enthalpy of vaporization @ 0C
        
    def es(self,T_K,a_param,b_param):
        """
        Parameters
        ----------
        T_K : temperature [K]
        a_param : The constant "a" [Pa] determined in the Problem 2.2
        b_param : The constant "b" [K] determined in the Problem 2.2

        Returns
        -------
        saturarion vapor pressure (es) [Pa]

        """
        import math
        val= a_param*math.exp(-b_param/T_K)
        return val
        
        
        
    
        
        
        
        
        
# -*- coding: utf-8 -*
"""
Created on Thu Mar 17 19:00:56 2022

@author: Moteki
"""
def miescat(wl_0,m_m,d_p,m_p_real,m_p_imag,dens_p=1.0,nang=3) :
    """ function miescat
    
    Calculating light scattering properties of single homogeneous sphere based
    on the mathematical formulations of [Bohren and Huffman 1983, Absorption and Scatteing
    of Light by Small Particles] (BH83), and [Fu and Sun 2001, Mie theory for
    light scattering by a spherical particle in an absorbing medium, Appl.Opt.
    40, 1354-1361] (FS01).
    
    ---Theoretical Assumptions---
    1.Gaussian unit is employed for mathematical expressions
    2.Surrounding medium must be nonabsorbing but can be magnetic (nonabsorbing assumption is necessary to define Qsca and Qabs without ambiguity)
    3.Particle can be absorbing and magnetic
    
    ---Computational Assumptions----
    1.Number of terms in Vector Spherical Wave Function (VSWF) expansion 'nstop' is determined as
    nstop=floor(x+4*x^0.3333+2) according to BH83
    where x is the size parameter defined below
    -----------------------------
    
    ---INPUT ARGUMENTS---
    wl_0 : wavelength in vacuum (=c_light/w) [m]
    m_m : refractive index of medium (real number)
    d_p: particle diameter [m]
    m_p_real : real part of the refractive index of particle (real number)
    m_p_imag : imaginary part of the refractive index of particle (real number)
    dens_p: particle density [g/cm3]
    nang: number of grid of scattering angle between 0-180 deg
    ---------------------
    
    m_m: refractive index of medium (real number)
    m_p: complex refractive index of particle m=n+ik
    m_r: relative complex refractive index of particle (=m_p/m_m)
    k_m: wavenumber in medium (real number)
    k_p: complex wavenumber in particle
    x: size parameter of particle with respect to the surrounding medium (=pi*d_p*m_m/wl_0)
    
    ---OUTPUTS---
    Qsca:  scattering efficiency := Csca/(pi*radius^2)
    Qext:  extinction efficiency := Cext/(pi*radius^2)
    Qabs:  absorption efficiency := Cabs/(pi*radius^2)
    S1(1:nang): (2,2) element of the 2*2 amplitude scattering matrix defined as BH83, Eq.3.12
    S2(1:nang): (1,1) element of the 2*2 amplitude scattering matrix defined as BH83, Eq.3.12
    MSC: scattering cross section per unit particle mass (mass scattering cross section) [m2/g]
    MEC: extinction cross section per unit particle mass (mass extinction cross section) [m2/g]
    MAC: absorption cross section per unit particle mass (mass absorption cross section) [m2/g]
    """

    import numpy as np

    m_p= m_p_real+1j*m_p_imag
    
    nang= int(nang)
    
    k0= 2*np.pi/wl_0
    k= m_m*k0
    x= k*d_p/2
    m_r= m_p/m_m

    nstop= int(np.floor(abs(x+4*x**0.3333+2))) # number of expansion terms for partial wave coefficients (BH83)
  
    y= m_r*x
    nmx= int(np.floor(max(nstop,abs(y))+15))
    
    #DD= logarithmic_derivative(m_r*x,nstop)  # BH83
    DD= np.zeros(nmx+1)+1j*np.zeros(nmx+1)  
    for n in range(nmx,0,-1):
        DD[n-1]= n/y-1/(DD[n]+n/y)
    DD= DD[0:nstop+1]
    
    # Reccati-Bessel function of first kind :=x*j(x)
    #psi= reccati_bessel_psi_dw(x,nstop,m_r)  # downward recurrence scheme (Mishchenko et al. 2002) psi[0], ..., psi[nstop]
    R= np.zeros(nmx+1) # R(n):=PSI(n)/PSI(n-1)
    R[nmx]=x/(2*nmx+1) # starting value of downward recurrence
    for n in range(nmx-1,-1,-1):
        R[n]=1/((2*n+1)/x-R[n+1]) # R(n) := Rn
    psi= np.zeros(nstop+1)
    psi[0]= R[0]*np.cos(x)
    for n in range(1,nstop+1):
        psi[n]= R[n]*psi[n-1]
    
    # Reccati-Bessel function of second kind :=x*y(x)
    #chi= reccati_bessel_chi(x,nstop)  #  chi[0], ..., chi[nstop]
    chi= np.zeros(nstop+1)
    chi[0]= -np.cos(x)
    chi[1]= (1/x)*chi[0]-np.sin(x)
    for n in range(2,nstop+1):
        chi[n]= ((2*n-1)/x)*chi[n-1]-chi[n-2]
    
    # Reccati-Bessel function of third kind := x*(j(x)+iy(x))
    xi = psi+1j*chi # xi[0], ..., xi[nstop]
    
    # Evaluations of partial wave coefficients a and b defined by BH83, Eqs.4.56-4.57 
    a= np.zeros(nstop+1)+1j*np.zeros(nstop+1) # define a[0:nstop]
    b= np.zeros(nstop+1)+1j*np.zeros(nstop+1) # define b[0:nstop]
    for n in range(1,nstop+1):
        a[n]=((DD[n]/m_r+n/x)*psi[n]-psi[n-1])/((DD[n]/m_r+n/x)*xi[n]-xi[n-1]) # BH83, Eq.4.88
        b[n]=((m_r*DD[n]+n/x)*psi[n]-psi[n-1])/((m_r*DD[n]+n/x)*xi[n]-xi[n-1]) # BH83, Eq.4.88
    
    fn1= np.zeros(nstop+1)
    fn2= np.zeros(nstop+1)
    sg= np.zeros(nstop+1)
    for n in range(1,nstop+1):
        fn1[n]=(2*n+1)
        fn2[n]=(2*n+1)/(n*(n+1))
        sg[n]=(-1)**n

    Qsca= (2/x**2)*np.sum(fn1*(np.abs(a)**2+np.abs(b)**2)) # BH83, Eq.4.61
    Qext= (2/x**2)*np.sum(fn1*np.real(a+b)) # BH83, Eq.4.62
    Qabs= Qext-Qsca
    Qback= 1/x**2*abs(sum(fn1*sg*(a-b)))**2
    
    Qsca= np.real(Qsca)
    Qext= np.real(Qext)
    Qabs= np.real(Qabs)
    Qback= np.real(Qback)

    theta= np.linspace(0,np.pi,nang) 
    pie= np.zeros((nstop+1,nang))
    tau= np.zeros((nstop+1,nang))
    
    mu= np.cos(theta)
    pie[1,:]= 1
    pie[2,:]= 3*mu*pie[1,:]
    tau[1,:]= mu*pie[1,:]
    tau[2,:]= 2*mu*pie[2,:]-3*pie[1,:]
    for n in range(3,nstop+1):
        pie[n,:]= ((2*n-1)/(n-1))*mu*pie[n-1,:]-(n/(n-1))*pie[n-2,:]
        tau[n,:]= n*mu*pie[n,:]-(n+1)*pie[n-1,:]
    
    S1= np.zeros(nang)+1j*np.zeros(nang)
    S2= np.zeros(nang)+1j*np.zeros(nang)
    for j in range(0,nang):
        S1[j]= sum(fn2*(a*pie[:,j]+b*tau[:,j])) # BH83, Eq.4.74
        S2[j]= sum(fn2*(a*tau[:,j]+b*pie[:,j])) # BH83, Eq.4.74

    #definition of scattering amplitude matrix in Mishchenko
    S11 = S2/(-1j*k)
    S22 = S1/(-1j*k)

    Csca= (np.pi*(d_p**2)/4)*Qsca  # [m2]
    Cext= (np.pi*(d_p**2)/4)*Qext  # [m2]
    Cabs= (np.pi*(d_p**2)/4)*Qabs  # [m2]

    v_p = (np.pi/6)*(d_p*1e2)**3  # [cm3]
    mass_p= v_p*dens_p # [g]
    MSC= Csca/mass_p
    MEC= Cext/mass_p
    MAC= Cabs/mass_p 

    return Qsca,Qext,Qabs,Qback,S11,S22,MSC,MEC,MAC

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.colors import LogNorm
from multiprocessing import Process
import os
import sys
import time
from matplotlib.colors import LogNorm
from matplotlib.offsetbox import AnchoredText
from scipy.stats import maxwell
from scipy.interpolate import RegularGridInterpolator


import Project_VF as Sim

#the different case of external electric field or 
def E_radial(x,y,iteration_count,timestep):
    norm=np.sqrt(x**2+y**2+z**2)
    Ex=-5e-5*x/norm
    Ey=-5e-5*y/norm
    return(np.array([Ex,Ey]))

def a_zero(x,y,vx,vy,timestep,iteration_count):
    return(0)

def B(x,y,vx,vy,iteration_count,timestep):
    B=5e-8
    F_x=vy*B
    F_y=vx*B
    return(constants.e/constants.m_e*np.array([F_x,F_y]))

def E_0(x,y,iteration_count,timestep):
    return(np.array([0,0]))

def E_x(x,y,iteration_count,timestep):
    Ex=5e-6
    Ey=0 
    return(np.array([Ex,Ey]))

def E_time(x,y,iteration_count,timestep):
    omega=np.pi/100
    k=np.pi/2
    norm=np.sqrt(x**2+y**2)
    if iteration_count>=1000:
        Ex=0
        Ey=5e-5*np.cos(k*x-omega*iteration_count)
    else :
        Ex=0
        Ey=0
    E=[Ex,Ey]
    return(np.array(E))
    return(np.array([Ex,Ey]))
    
#create the global variable of the simulation
def general_variables():
    global density2deg
    density2deg=1e3
    global relativeeps
    relativeeps=1
    global minimalresidue
    minimalresidue=7.5e-6
    global xmin
    xmin=-1
    global ymin
    ymin=-1
    global ymax
    ymax=1
    global xmax
    xmax=1
    global quantity
    quantity=250
    global timemax
    timemax=25000
    global mean_free_time
    mean_free_time=5e-4
    global temperature
    temperature=0.2
    global initial_velocity
    initial_velocity="rest"
    global timestep
    timestep=5e-6
    
    

#The different case/testing/results
def sweep_z(zmax,N):
    mean_free_time=1e-3 # we increase it, to speed up the process (we can bigger timestep)
    timestep=1e-5
    minimalresidue=2e-5
    quantity=50
    folder="SweepZ"
    V=[]
    global z
    z_list=np.linspace(0.001,zmax,num=N)
    electrongas=Sim.Simulation_Electron(folder,density2deg,mean_free_time,relativeeps,timestep,minimalresidue,temperature,E_radial,a_zero)
    for i in z_list:
        z=i
        electrongas.createcloudelectron(xmin, ymin, xmax, ymax, quantity, initial_velocity)
        print("Starting new run")
        electrongas.lauch(5000,animation=False)
        V.append(electrongas.Electrical_Potential())
    x=np.linspace(xmin,xmax,num=5*len(electrongas.x))
    y=np.linspace(ymin,ymax,num=5*len(electrongas.x))
    V=np.array(V)
    points=(z_list,x,y)
    print("Start spline")
    V_Estimation=RegularGridInterpolator(points,np.array(V),method="cubic")
    
    minimalresidue=1e-5
    quantity=250
    timestep=1e-6
    mean_free_time=1e-3

    return(V_Estimation)

def Spline_z(z,V_Estimation):
    N=500
    x=np.linspace(xmin,xmax,num=N)
    y=np.linspace(ymin,ymax,num=N)
    X,Y=np.meshgrid(x,y)
    V=X*0
    for j in range(0,N):
        V[:,j]=V_Estimation((z,X[:,j],Y[:,j]))
    May_be_electron=np.abs(V)>np.mean(np.abs(V))*5
    plt.pcolor(x,y,May_be_electron)
    plt.xlabel("X [µm]")
    plt.ylabel("Y [µm]")
    plt.show()
    
def linear_mouvement():
    folder="Linear_Mouvement"
    electrongas=Sim.Simulation_Electron(folder,density2deg,mean_free_time,relativeeps,timestep,minimalresidue,temperature,E_x,a_zero)
    electrongas.createcloudelectron(xmin, ymin, xmax, ymax, quantity, initial_velocity)
    electrongas.lauch(300)

def accumulation():
    folder="Linear_Mouvement"
    electrongas=Sim.Simulation_Electron(folder,density2deg,mean_free_time,relativeeps,timestep,minimalresidue,temperature,E_x,a_zero,boundary="momentum conserve")
    electrongas.createcloudelectron(xmin, ymin, xmax, ymax, quantity, initial_velocity)
    electrongas.lauch(timemax)

def radial():
    timestep=1e-6
    global z
    z=0
    folder="Radial"
    electrongas=Sim.Simulation_Electron(folder,density2deg,mean_free_time,relativeeps,timestep,minimalresidue,temperature,E_radial,a_zero)
    electrongas.createcloudelectron(xmin, ymin, xmax, ymax, quantity, initial_velocity)
    electrongas.lauch(timemax)

def Nothing():
    quantity=400
    minimalresidue=1e-6
    folder="Nothing"
    electrongas=Sim.Simulation_Electron(folder,density2deg,mean_free_time,relativeeps,timestep,minimalresidue,temperature,E_0,a_zero)
    electrongas.createcloudelectron(xmin, ymin, xmax, ymax, quantity, initial_velocity)
    electrongas.lauch(timemax,animation="Timing",animation_timing=100)
    return(electrongas)


def Phase_Lag():
    mean_free_time=1e-3
    minimalresidue=0.1e-5
    timestep=5e-6
    folder="Phase_Lag"
    electrongas=Sim.Simulation_Electron(folder,density2deg,mean_free_time,relativeeps,timestep,minimalresidue,temperature,E_time,a_zero)
    electrongas.createcloudelectron(xmin, ymin, xmax, ymax, quantity, initial_velocity)
    electrongas.lauch(3000,animation=False,phase_lag=True)
    mean_free_time=1e-6
    timestep=1e-5
    minimalresidue=1e-5


def leapfrog():
    timestep=5e-5
    quantity=400
    minimalresidue=1e-6
    folder="Leapfrog"
    electrongas=Sim.Simulation_Electron(folder,density2deg,mean_free_time,relativeeps,timestep,minimalresidue,temperature,E_0,a_zero,ode="Leapfrog")
    electrongas.createcloudelectron(xmin, ymin, xmax, ymax, quantity, initial_velocity)
    electrongas.lauch(timemax,animation="Timing",animation_timing=100)


    
if __name__ == '__main__':
    
    """
    Just take a case and run it.
    The only one that need more preparation is the sweep in z direction which is in comment below.
    """
    
    general_variables()
    linear_mouvement()
    
    """
    V=sweep_z(5,50)
    #Spline_z(2,V) #replace 2 for any z value that you want
    """
    


    
    
    
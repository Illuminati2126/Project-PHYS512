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

class Simulation_Electron :
    def __init__(self,folder,density,mean_free_path,relativeeps,timestep,minimalresidue, temperature, External_Electric_Field ): #information about the 2DEG
        self.folder=folder
        self.temperature=temperature
        self.density=density #um^-2
        self.minimalresidue=minimalresidue
        self.mean_free_path=mean_free_path 
        #self.boundary="momentumconserve"
        self.boundary="momentum conserve"
        self.relativeeps=relativeeps
        self.timestep=timestep
        self.counting=0
        self.External_E=External_Electric_Field
        
        
    def createcloudelectron(self,xmin,ymin,xmax,ymax,z,quantity,initialspeed):
        self.iterationcount=0
        self.xmin=xmin
        self.ymin=ymin
        self.xmax=xmax
        self.ymax=ymax
        A=(abs(xmax)+abs(xmin))*(abs(ymax)+abs(ymin)) #area
        self.quantity=quantity
        self.charge=constants.e*A*self.density/quantity # artificly augmente the charge by particle
        #tempox=np.linspace(xmin,xmax,num=round(np.sqrt(self.quantity))) #generate the position of them in x
        #tempoy=np.linspace(ymin,ymax,num=round(np.sqrt(self.quantity)))
        #self.x,self.y=np.meshgrid(tempox,tempoy) #position of the electron
        self.x=random.uniform(low=xmin,high=xmax,size=(self.quantity))
        self.y=random.uniform(low=ymin,high=ymax,size=(self.quantity))
        if initialspeed=="rest":
            self.vx=np.zeros(len(self.x))
            self.vy=np.zeros(len(self.y))
        else: #JM:Should implement random distribution (e.g. Maxxwell)
            self.vx=-initialspeed+random.random(len(self.x))*2*initialspeed #uniform distribution around the 0 with a range of initialspeed
            self.vy=-initialspeed+random.random(len(self.y))*2*initialspeed

    def boundarycondition(self):
        if self.boundary=="periodic":
            while np.any(self.x>=self.xmax) or np.any(self.y>=self.ymax) or np.any(self.x<=self.xmin) or np.any(self.y<=self.ymin):
                index=np.nonzero(self.x>=self.xmax)
                self.x[index]=self.xmin+abs(self.x[index]-self.xmax)
                index=np.nonzero(self.y>=self.ymax)
                self.y[index]=self.ymin+abs(self.y[index]-self.ymax)
                index=np.nonzero(self.x<=self.xmin)
                self.x[index]=self.xmax-abs(-self.x[index]+self.xmin)
                index=np.nonzero(self.y<=self.ymin)
                self.y[index]=self.ymax-abs(-self.y[index]+self.ymin)

        elif self.boundary=="momentum conserve":
            index=np.nonzero(self.x>=self.xmax)
            self.x[index]=self.xmax
            self.vx[index]=-self.vx[index]
            index=np.nonzero(self.y>=self.ymax)
            self.y[index]=self.ymax
            self.vy[index]=-self.vy[index]
            index=np.nonzero(self.x<=self.xmin)
            self.x[index]=self.xmin
            self.vx[index]=-self.vx[index]
            index=np.nonzero(self.y<=self.ymin)
            self.y[index]=self.ymin
            self.vy[index]=-self.vy[index]
    
    def InternalElectrical(self,x,y):# electrical field generate by the cloud of electron
        Ex=0
        Ey=0
        with np.errstate(invalid='raise'): # just to catch division by 0 and avoid inf
            #direct radius without any funky boundary
            xd=(x-self.x)
            yd=(y-self.y)
            
            Lx=self.xmax-self.xmin
            Ly=self.ymax-self.ymin
            index=(Lx)/2<np.abs(xd)
            xd[index]=(Lx-np.abs(xd[index]))*-(xd[index]/np.abs(xd[index]))
            index=(Ly)/2<np.abs(yd)
            yd[index]=(Ly-np.abs(yd[index]))*-(yd[index]/np.abs(yd[index]))
            
            
            distance=(np.sqrt(xd**2+yd**2))
            
            index=np.nonzero(distance==0) # two particle at the same position
            distance[index]=1 #the xd,yd,zd must be equal to 0, so they make E=0. It is only done to avoid division by 0
            try:
                Ex=xd*distance**-3
                Ey=yd*distance**-3
            except : 
                print("There is a problem. Go see the InternalElectrical")
                pass
        
        
            index=np.logical_not(np.logical_and(xd==0,yd==0))
            sign=xd[index]
            zeroindex=np.logical_not(xd[index]==0)
            sign[zeroindex]=(xd[zeroindex]/np.abs(xd[zeroindex]))
            xd[index]=(Lx-np.abs(xd[index]))*-(sign)
            zeroindex=np.logical_not(xd[index]==0)
            sign=yd[index]
            sign[zeroindex]=(yd[zeroindex]/np.abs(yd[zeroindex]))
            yd[index]=(Ly-np.abs(yd[index]))*-(sign)
            
            distance=(np.sqrt(xd**2+yd**2))
            
            index=np.nonzero(distance==0) # two particle at the same position
            distance[index]=1 #the xd,yd,zd must be equal to 0, so they make E=0. It is only done to avoid division by 0
            try:
                Ex=Ex+xd*distance**-3
                Ey=Ey+yd*distance**-3
            except : 
                print("There is a problem. Go see the InternalElectrical")
                pass
        
        E=np.array([np.sum(Ex),np.sum(Ey)])*self.charge/(4*np.pi*constants.epsilon_0*(self.relativeeps))
        return(E)
    
    def flexible(self):
        #Adapt the limit of the simulation to follow the particles
        if np.min(self.x)<self.xmin:
            self.xmin=np.min(self.x)
        if np.min(self.y)<self.ymin:
            self.ymin=np.min(self.y)
        if np.max(self.y)>self.ymax:
            self.ymax=np.max(self.y)
        if np.max(self.x)>self.xmax:
            self.xmax=np.max(self.x)

    def diffusion(self): #JM: TO MODIF FOR EXPOTENTIAL DAMPING
        self.vx=self.vx*np.exp(-self.timestep/self.mean_free_path)
        self.vy=self.vy*np.exp(-self.timestep/self.mean_free_path)
    
    def iteration(self):
        self.iterationcount=self.iterationcount + 1
        ax=np.zeros(len(self.x))
        ay=np.zeros(len(self.x))
        for i in range(0,len(self.x)):
            ax[i], ay[i] = constants.e/constants.m_e*(self.InternalElectrical(self.x[i],self.y[i])+self.External_E(self.x[i],self.y[i],self.iterationcount,self.timestep))
                
        deltavx=ax*self.timestep
        deltavy=ay*self.timestep
        residuex=self.timestep*(self.vx+ax*self.timestep/2)
        residuey=self.timestep*(self.vy+ay*self.timestep/2)
        self.x=self.x+residuex
        self.y=self.y+residuey
        self.vx=self.vx+deltavx
        self.vy=self.vy+deltavy
        self.residuex=np.mean(np.abs(residuex)) 
        self.residuey=np.mean(np.abs(residuey))
        
    def show_electron(self,save):
        plt.pause(0.001)
        plt.clf()
        plt.xlim(self.xmin,self.xmax)
        plt.ylim(self.ymin,self.ymax) 
        plt.xlabel("X [µm]")
        plt.ylabel("Y [µm]")
        plt.scatter(self.x,self.y,color="black",s=5, zorder=200)
        if save==True:
            SaveFile = "Figure" + "_step_" + str(self.iterationcount) + ".png" 
            plotname=os.path.join(os.getcwd(),self.folder,SaveFile)
            plt.savefig(plotname)
        plt.show()
        #plt.close()
    """
    def show_Electrical_Field(self,save):
        plt.clf()
        plt.xlim(self.xmin,self.xmax)
        plt.ylim(self.ymin,self.ymax) 
        plt.xlabel("X [µm]")
        plt.ylabel("Y [µm]")
        
        xvec=np.linspace(self.xmin,self.max,num=2*len(self.x))
        yvec=np.linspace(self.xmin,self.max,num=2*len(self.x))
        Ex=np.zeros((len(xvec),len(xevc)))
        Ey=np.zeros((len(xvec),len(xevc)))
        X,Y= np.meshgrid(xvec,yvec)
        for i in range(0,len(X)):
            for j in range(0,len(Y)):
                Ex[i,j], Ey[i,j], =(self.InternalElectrical(self.x[i,j],self.y[i,j],self.z))
        normE=np.sqrt(Ex**2+Ey**2)
        
        #plt.pcolor(xvec,yvec,normE,zorder=0,norm=LogNorm(vmin=abs(norm.min())))
        plt.colorbar(orientation="horizontal",label="Norm of the electrical field [kg$\cdot$µm/(A$\cdot s^{3}$)]")
        plt.streamplot(xvec,yvec,Ex,Ey,density=1, zorder=25,color="green")# vector field of the plot
        if save==True:
            SaveFile = "Figure" + "_step_" + str(self.iterationcount) + ".png" 
            plotname=os.path.join(os.getcwd(),self.folder,SaveFile)
            plt.savefig(plotname)
        plt.show()
        plt.close()
    """

    def lauch(self,maxtime):
        self.show_electron(True)
        while self.iterationcount<maxtime :
            self.iteration()
            self.boundarycondition()
            self.diffusion()
            self.flexible()
            self.show_electron(False)
            if self.minimalresidue>self.residuex and self.minimalresidue>self.residuey and self.iterationcount>100:
                print("We are now into a steady situation.")
                break
            if (self.iterationcount%50)==0:
                print("Iteration:"+str(self.iterationcount))

"""
To do :
    - initial condition to modified to have random distribution
    - add the exp damping term
    - add external electrical field
Bonus:
    - create code to have video from the thounsand of png (and delete them after use)
    - time dependant electrical field
    - optimize the code (electrical field calculation), multiprocessing and python optimization
    - Correlation function (FFT)
    - add extremely funky electric potential
Question:
    - should we modif the code to instead use electric potential instead of electric field ? It would let us introduce a infinite/high potential at the boundary to avoid the electron to accumulate to the limit

"""
def E_ext(x,y,iteration,timestep):
    Ex=1e-5
    Ey=0
    return(np.array([Ex,Ey]))


if __name__ == '__main__':
    folder="Picture"
    density2deg=1e3
    mean_free_path=1e-3
    relativeeps=1
    timestep=5e-6
    minimalresidue=8e-6
    temperature=0.2
    
    xmin=-2
    ymin=-2
    xmax=2
    ymax=2
    quantity=300
    initial_velocity="rest"
    z=1e-12
    timemax=1000000
    electrongas=Simulation_Electron(folder,density2deg,mean_free_path,relativeeps,timestep,minimalresidue,temperature,E_ext)
    electrongas.createcloudelectron(xmin, ymin, xmax, ymax, z, quantity, initial_velocity)
    print("Starting the simulation")
    electrongas.lauch(timemax)


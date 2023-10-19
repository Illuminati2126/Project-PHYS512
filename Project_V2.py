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
    def __init__(self,folder,density,mean_free_path,relativeeps,timestep,minimalresidue, temperature ): #information about the 2DEG
        self.folder=folder
        self.temperature=temperature
        self.density=density #um^-2
        self.minimalresidue=minimalresidue
        self.mean_free_path=mean_free_path 
        self.boundary="momentumconserve"
        self.relativeeps=relativeeps
        self.timestep=timestep
        self.counting=0
        
    def createcloudelectron(self,xmin,ymin,xmax,ymax,z,quantity,initialspeed):
        self.iterationcount=0
        self.xmin=xmin
        self.ymin=ymin
        self.xmax=xmax
        self.ymax=ymax
        A=(abs(xmax)+abs(xmin))*(abs(ymax)+abs(ymin)) #area
        if quantity=="real": #DANGER IT WILL BE BIG AND EXTREMELY HARD TO RUN, IF NOT IMPOSSIBLE TO RUN
            self.quantity=A*self.density
            self.charge=constants.e
        else:
            self.quantity=quantity
            self.charge=constants.e*A*self.density/quantity
        #tempox=np.linspace(xmin,xmax,num=round(np.sqrt(self.quantity))) #generate the position of them in x
        #tempoy=np.linspace(ymin,ymax,num=round(np.sqrt(self.quantity)))
        #self.x,self.y=np.meshgrid(tempox,tempoy) #position of the electron
        self.x=random.uniform(low=xmin,high=xmax,size=(self.quantity,self.quantity))
        self.y=random.uniform(low=ymin,high=ymax,size=(self.quantity,self.quantity))
        self.z=z
        if initialspeed=="rest":
            self.vx=np.zeros((len(self.x),len(self.x)))
            self.vy=np.zeros((len(self.y),len(self.y)))
        else: #JM:Should implement random distribution (e.g. Maxxwell)
            self.vx=-initialspeed+random.random((len(self.x),len(self.x)))*2*initialspeed #uniform distribution around the 0 with a range of initialspeed
            self.vy=-initialspeed+random.random((len(self.y),len(self.y)))*2*initialspeed

    def boundarycondition(self):
        debug=False
        #force the electron to never quit the testing region
        indexx,indexy=np.nonzero(self.x>self.xmax)
        if debug==True and len(indexx)>0:
            print("Charge carrier over xmax for those index:")
            print("i:"+str(indexx)+" and j:" +str(indexy))
        for i in indexx:
            for j in indexy:
                if self.boundary=="momentumconserve":
                    self.x[indexx,indexy]=self.xmax
                    self.vx[indexx,indexy]=-self.vx[indexx,indexy]
                elif self.boundary=="stop":
                    self.x[indexx,indexy]=self.xmax
                    self.vx[indexx,indexy]=0
                elif self.boundary=="periodic":
                    self.x[indexx,indexy]=self.xmin+abs(self.x[indexx,indexy]-self.xmax)
                    self.vx[indexx,indexy]=self.vx[indexx,indexy]

        indexx,indexy=np.nonzero(self.y>self.ymax)
        if debug==True and len(indexx)>0:
            print("Charge carrier over ymax for those index:")
            print("i:"+str(indexx)+" and j:" +str(indexy))
        for i in indexx:
            for j in indexy:
                if self.boundary=="momentumconserve":
                    self.y[indexx,indexy]=self.ymax
                    self.vy[indexx,indexy]=-self.vy[indexx,indexy]
                elif self.boundary=="stop":
                    self.y[indexx,indexy]=self.ymax
                    self.vy[indexx,indexy]=0
                elif self.boundary=="periodic":
                    self.y[indexx,indexy]=self.ymin+abs(self.y[indexx,indexy]-self.ymax)
        
        indexx,indexy=np.nonzero(self.x<self.xmin)
        if debug==True and len(indexx)>0:
            print("Charge carrier under xmin for those index:")
            print("i:"+str(indexx)+" and j:" +str(indexy))
        for i in indexx:
            for j in indexy:
                if self.boundary=="momentumconserve":
                    self.x[indexx,indexy]=self.xmin
                    self.vx[indexx,indexy]=-self.vx[indexx,indexy]
                elif self.boundary=="stop":
                    self.x[indexx,indexy]=self.xmin
                    self.vx[indexx,indexy]=0
                elif self.boundary=="periodic":
                    self.x[indexx,indexy]=self.xmax-abs(-self.x[indexx,indexy]+self.xmin)

        indexx,indexy=np.nonzero(self.y<self.ymin)
        if debug==True and len(indexx)>0:
            print("Charge carrier under ymax for those index:")
            print("i:"+str(indexx)+" and j:" +str(indexy))
        for i in indexx:
            for j in indexy:
                if self.boundary=="momentumconserve":
                    self.y[indexx,indexy]=self.ymin
                    self.vy[indexx,indexy]=-self.vy[indexx,indexy]
                elif self.boundary=="stop":
                    self.y[indexx,indexy]=self.ymin
                    self.vy[indexx,indexy]=0
                elif self.boundary=="periodic":
                    self.y[indexx,indexy]=self.ymax-abs(-self.y[indexx,indexy]+self.ymin)
    
    def InternalElectrical(self,x,y,z):# electrical field generate by the cloud of electron
        Ex=0
        Ey=0
        Ez=0
        with np.errstate(invalid='raise'):
            xd=(x-self.x)
            yd=(y-self.y)
            zd=(z-self.z)
            distance=(np.sqrt(xd**2+yd**2+zd**2))
            index=np.nonzero(distance==0)
            distance[index]=1 
            try:
                Ex=xd*distance**-3
                Ey=yd*distance**-3
                Ez=zd*distance**-3
            except : 
                print("There is a problem. Go see the InternalElectrical")
                pass
        #print("internal")
        #print(np.array([np.sum(Ex),np.sum(Ey),np.sum(Ez)]))
        E=np.array([np.sum(Ex),np.sum(Ey),np.sum(Ez)])*self.charge/(4*np.pi*constants.epsilon_0*(self.relativeeps))
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
        ax=np.zeros((len(self.x),len(self.y)))
        ay=np.zeros((len(self.x),len(self.y)))
        az=np.zeros((len(self.x),len(self.y)))
        for i in range(0,len(self.x)):
            for j in range(0,len(self.y)):
                ax[i,j], ay[i,j], az[i,j] = constants.e/constants.m_e*(self.InternalElectrical(self.x[i,j],self.y[i,j],self.z))
                
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
        print("Average mouvement is ",np.sqrt(self.residuex**2+self.residuey**2))

        
    def show_electron(self,save):
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
        plt.close()
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
            if self.iterationcount%250:
                print("Iteration:"+str(self.iterationcount))
            self.iteration()
            self.boundarycondition()
            self.diffusion()
            self.flexible()
            self.show_electron(True)
            if self.minimalresidue>self.residuex and self.minimalresidue>self.residuey and self.iterationcount>100:
                print("We are now into a steady situation.")
                break

"""
To do :
    - boundary condition to update
    - initial condition to modified to have random distribution
    - add the exp damping term
    - addapt diffusion term to be a scattering function
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



if __name__ == '__main__':
    folder="test"
    density2deg=1e3
    mean_free_path=1e-3
    relativeeps=1
    timestep=1e-4
    minimalresidue=1e-6
    temperature=0.2
    
    xmin=-2
    ymin=-2
    xmax=2
    ymax=2
    quantity=5
    initial_velocity="rest"
    z=1e-12
    timemax=5000
    electrongas=Simulation_Electron(folder,density2deg,mean_free_path,relativeeps,timestep,minimalresidue,temperature)
    electrongas.createcloudelectron(xmin, ymin, xmax, ymax, z, quantity, initial_velocity)
    print("Starting the simulation")
    electrongas.lauch(timemax)


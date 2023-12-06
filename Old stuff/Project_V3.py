import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy import constants
import os
from scipy.stats import maxwell
from scipy.interpolate import RegularGridInterpolator
import matplotlib.colors as colors




class Simulation_Electron :
    def __init__(self,folder,density,mean_free_time,relativeeps,timestep,minimalresidue, temperature,External_Electric_Field,External_Acceleration,ode="Euler",boundary="periodic"):
        """
        Initialize the generate information of the simulation.

        Parameters
        ----------
        folder : str
            Name of the folder where we save any file.
        density : double
            Density of electron of the 2DEG in um^-2.
        mean_free_time : double
            Mean free time of the electron inside the 2DEG in picosecond.
        relativeeps : double
            The relative permativity of the 2DEG.
        timestep : double
            The time step used in the simulation in picosecond.
        minimalresidue : double
            The minimal residue that is consider as a non-steady state.
        temperature : double
            The temperature of the 2DEG.
        External_Electric_Field : Function
            The function used to represent the external electric force.
        External_Acceleration : Function
            The function used to represent the external force acting on the electron
        ode : double, optional
            The numerical scheme used to the solve the simulation, either Euler or Leapfrog. The default is "Euler". 
        boundary : double, optional
            The type of boundary used for the simulation, either "momentum conserve" for reflective boundary or "periodic" for the periodic boundary. The default is "periodic".

        Returns
        -------
        None.

        """
        self.folder=folder 
        self.density=density 
        self.minimalresidue=minimalresidue 
        self.mean_free_path=mean_free_path 
        self.boundary=boundary 
        self.relativeeps=relativeeps 
        self.timestep=timestep 
        self.External_E=External_Electric_Field  
        self.External_a=External_Acceleration 
        self.ode=ode #
        self.random=maxwell() 
        self.variance=np.sqrt(constants.m_e/(constants.Boltzmann*temperature))

        
    def createcloudelectron(self,xmin,ymin,xmax,ymax,quantity,initialspeed): 
        """
        Create/regenerate the cloud of electron in the 2DEG
        Parameters
        ----------
        xmin : double
            Minimal x.
        ymin : double
            Maximal x.
        xmax : double
            Minimal y.
        ymax : double
            Maximal y.
        quantity : int
            The quantity of electron simulated.
        initialspeed : str
            The type of initial condition for the velocity. Either at rest or following Maxwell-Bolztmann distribution

        Returns
        -------
        None.

        """
        
        self.iterationcount=0
        self.xmin=xmin
        self.ymin=ymin
        self.xmax=xmax
        self.ymax=ymax
        A=(abs(xmax)+abs(xmin))*(abs(ymax)+abs(ymin)) #area
        self.quantity=quantity
        self.charge=constants.e*A*self.density/quantity # artificly augmente the charge by particle, so the total charge of the 2DEG stay realistic with the number of particle that we chosen 
        #The initial position of the electron are chosen with a uniform distribution
        self.x=random.uniform(low=xmin,high=xmax,size=(self.quantity)) 
        self.y=random.uniform(low=ymin,high=ymax,size=(self.quantity))
        if initialspeed=="rest":
            self.vx=np.zeros(len(self.x))
            self.vy=np.zeros(len(self.y))
        else: #give just a little bit of velocity to the electron. Should instead start from Max.Bolzt. distribution to be more realistic.
            self.vx=-initialspeed+random.random(len(self.x))*2*initialspeed #uniform distribution around the 0 with a range of initialspeed
            self.vy=-initialspeed+random.random(len(self.y))*2*initialspeed

    def boundarycondition(self):
        """
        Used to force the boundary condition.

        Returns
        -------
        None.

        """
        
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
                 
                        
    def InternalElectrical(self,x,y):
        """
        Find the electrical field generate by electron in the 2DEG for the x and y position

        Parameters
        ----------
        x : double (NOT A ARRAY)
            A x position.
        y : double (NOT A ARRAY)
            A y position.

        Returns
        -------
        np.array of Ex (double) and Ey (double).

        """
        
        Ex=0
        Ey=0
        with np.errstate(invalid='raise'): # just to catch division by 0 and avoid inf
            #direct radius without any funky boundary
            xd=(x-self.x)
            yd=(y-self.y)
            
            Lx=self.xmax-self.xmin # dimension of the 2DEG
            Ly=self.ymax-self.ymin
            
            # this is for taking the distance between the electron, but by passing by the other side (passing by the periodic boundary)
            # If the distance is inferior to L/2, so inferior to the classic method, then we take that distance instead 
            index=(Lx)/2<np.abs(xd)
            xd[index]=(Lx-np.abs(xd[index]))*-(xd[index]/np.abs(xd[index]))
            index=(Ly)/2<np.abs(yd)
            yd[index]=(Ly-np.abs(yd[index]))*-(yd[index]/np.abs(yd[index]))
            
            
            distance=(np.sqrt(xd**2+yd**2))
            
            index=np.nonzero(distance==0) # two particle at the same position
            distance[index]=1 #the xd,yd,zd must be equal to 0, so they make E=0 (we just ignore the electric field in that case). It is only done to avoid division by 0
            #we ignore it, because it is virtually impossible to really happen, the potential will be soo big that the electron will never be in the contact with the other electron. So, if it is happening it probrably some numerical error (the initial condition make them starting at the same place) and the momentum will push it away by itself. Thus, we just ignore it.
            try:
                Ex=xd*distance**-3
                Ey=yd*distance**-3
            except : 
                #this imply that measure put in place to avoid division by zero have fail and thus there is now a NaN or Inf in the V, so the simulation should be stop and the bug should be found
                print("There is a problem. Go see the InternalElectrical")
                pass
        E=np.array([np.sum(Ex),np.sum(Ey)])*self.charge/(4*np.pi*constants.epsilon_0*(self.relativeeps)) #sum of the contribution of each electron to the electric field and multiply by the proper constants 
        return(E)
    
    def flexible(self):
        """
        Adapt the limit of the simulation, if one particle go beyond the limit.
        In theory, it should never be used, but it is a great way to keep track of major bug in the boundary condition or if there is no boundary condition
        
        Returns
        -------
        None.

        """
        #Adapt the limit of the simulation to follow the particles
        if np.min(self.x)<self.xmin:
            self.xmin=np.min(self.x)
        if np.min(self.y)<self.ymin:
            self.ymin=np.min(self.y)
        if np.max(self.y)>self.ymax:
            self.ymax=np.max(self.y)
        if np.max(self.x)>self.xmax:
            self.xmax=np.max(self.x)

    def diffusion(self):
        """
        The diffusion of momentum inside the 2DEG, due to the scattering of the electron with the impurity and phonons.
        Based on the Drude model.
        Returns
        -------
        None.

        """
        self.vx=self.vx*np.exp(-self.timestep/self.mean_free_path)
        self.vy=self.vy*np.exp(-self.timestep/self.mean_free_path)
    
    def iteration(self):
        """
        Make one time iterations for all the electron based on the numerical initially chosen.

        Returns
        -------
        None.

        """
        self.iterationcount=self.iterationcount + 1
        ax=np.zeros(len(self.x))
        ay=np.zeros(len(self.x))
        if self.ode=="Euler":
            for i in range(0,len(self.x)):
                ax[i], ay[i] = self.External_a(self.x[i],self.y[i],self.vx[i],self.vy[i],timestep,self.iterationcount) + constants.e/constants.m_e*(self.InternalElectrical(self.x[i],self.y[i])+self.External_E(self.x[i],self.y[i],self.iterationcount,self.timestep)) #calculated the acceleration for each one of the electrons
                    
            deltavx=ax*self.timestep #residue for the velocity
            deltavy=ay*self.timestep
            residuex=self.timestep*(self.vx+ax*self.timestep/2) # how much each the electron will move, e.g. the residue
            residuey=self.timestep*(self.vy+ay*self.timestep/2)
            self.x=self.x+residuex #Change the position of each electron to their new positions
            self.y=self.y+residuey
            self.vx=self.vx+deltavx #Change the velocity of the electron, for their values
            self.vy=self.vy+deltavy
        elif self.ode=="Leapfrog":
            for i in range(0,len(self.x)):
                ax[i], ay[i] = self.External_a(self.x[i],self.y[i],self.vx[i],self.vy[i],timestep,self.iterationcount) + constants.e/constants.m_e*(self.InternalElectrical(self.x[i],self.y[i])+self.External_E(self.x[i],self.y[i],self.iterationcount,self.timestep)) #calculated the acceleration for each one of the electrons
            half_vx=self.vx+ax*self.timestep/2 # velocity for a half iterations
            residuex=half_vx*self.timestep  # the change of positions of the electrons
            self.x=residuex+self.x # Modify the position of electron for this iterations
            half_vy=self.vy+ay*self.timestep/2
            residuey=half_vy*self.timestep
            self.y=residuey+self.y
            for i in range(0,len(self.x)):
                ax[i], ay[i] =self.External_a(self.x[i],self.y[i],self.vx[i],self.vy[i],timestep,self.iterationcount) + constants.e/constants.m_e*(self.InternalElectrical(self.x[i],self.y[i])+self.External_E(self.x[i],self.y[i],self.iterationcount,self.timestep)) #Calculated the acceleration for this new position after this half iterations
            self.vx=half_vx+ax*self.timestep/2 #The new velocity after this FULL iterations
            self.vy=half_vy+ay*self.timestep/2
            
        self.residuex=np.mean(np.abs(residuex)) #Average residue of the electron
        self.residuey=np.mean(np.abs(residuey))
        
    def show_electron(self,save=False):
        """
        Figure of the position of the electron on the 2DEG.
        Made to be easily used as animation.

        Parameters
        ----------
        save : Bool, optional
            Do you want to save the figure representing the position of the electron. The default is False.

        Returns
        -------
        None.

        """
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
        
    def show_Electrical_Field(self,save=False):
        """
        Calculated the electrical field at y=0 for the internal electrical field and the external electrical field and generate a vector field plot of the Ex.
        It is used to show the phased lag between, the internal and external electrical field. 
        Is made to be on the form of an animation.
        
        Parameters
        ----------
        save : Bool, optional
            Do you want to the save the figure of the y-electrical field. The default is False.

        Returns
        -------
        None.

        """
        N=100
        plt.clf()
        plt.xlim(self.xmin,self.xmax)
        plt.ylim(self.ymin,self.ymax) 
        plt.xlabel("X [µm]")
        plt.ylabel("Y [µm]")
        xvec=np.linspace(self.xmin,self.xmax,num=N)
        other,E=self.External_E(xvec,0,self.iterationcount,self.timestep)
        plus=E>0
        minus=E<0
        plt.scatter(xvec[plus],0*xvec[plus],color="blue")
        plt.scatter(xvec[minus],0*xvec[minus],color="red")
        plt.scatter(self.x,self.y,s=5,color="black")
        
        if save==True:
            SaveFile = "Electrical Field"+ ".png" 
            plotname=os.path.join(os.getcwd(),self.folder,SaveFile)
            plt.savefig(plotname)
        plt.show()
        plt.pause(0.001)

    def Internal_Potential(self,x,y): 
        """
        The electric potential generated by the electron inside the 2DEG for the position x and y.

        Parameters
        ----------
        x : Double
            x position.
        y : Double
            y position.

        Returns
        -------
        None.

        """
        
        #same logic as calculating the internal electric field
        V=0
        with np.errstate(invalid='raise'): # just to catch division by 0 and avoid inf
            xd=(x-self.x) # distance between x and the x positon of the electrons
            yd=(y-self.y) # distance between y and the y positon of the electrons
            
            distance=(np.sqrt(xd**2+yd**2)) #distance between the position and the electrons
            
            index=np.nonzero(distance==0) # two particle at the same position
            distance[index]=1e99 #if distance is zero, then we ignore that contribution to the potential
            #we ignore it, because it is virtually impossible to really happen, the potential will be soo big that the electron will never be in the contact with the other electron. So, if it is happening it probrably some numerical error (the initial condition make them starting at the same place) and the momentum will push it away by itself. Thus, we just ignore it.
            try:
                V=distance**-2
            except : 
                #this imply that measure put in place to avoid division by zero have fail and thus there is now a NaN or Inf in the V, so the simulation should be stop and the bug should be found
                print("There is a problem. Go see the InternalPotential")
                pass
        V=np.sum(V)*self.charge/(4*np.pi*constants.epsilon_0*(self.relativeeps)) #add all the individual potential of each electron and multiply by the proper constants 
        return(-V)    

    def Electrical_Potential(self,save=False,show=False):
        """
        Find the electric potential insidie the 2DEG.

        Parameters
        ----------
        save : Bool, optional
            Do you want to save the figure of the electric potential of the 2DEG. The default is False.
        show : Bool, optional
            Do you want to see the figure of the electric potential of the 2DEG. The default is False.

        Returns
        -------
        np.array of double. Following the form of a meshgrid.

        """
        xvec=np.linspace(self.xmin,self.xmax,num=5*len(self.x))
        yvec=np.linspace(self.ymin,self.ymax,num=5*len(self.x))
        V=np.zeros((len(yvec),len(yvec)))
        X,Y= np.meshgrid(xvec,yvec) #meshgrid where we calculted the potential inside the 2DEG due to the electron inside the 2DEG (ignore the potential due to the external electric field)
        for i in range(0,len(X)):
            for j in range(0,len(Y)):
                V[i,j]= self.Internal_Potential(X[i,j],Y[i,j]) #evaluate the potential on the meshgrid
        
        if show: #show the potential field calculated
            plt.clf()
            plt.xlim(self.xmin,self.xmax)
            plt.ylim(self.ymin,self.ymax) 
            plt.xlabel("X [µm]")
            plt.ylabel("Y [µm]")
            
            plt.pcolor(xvec,yvec,V)
            
            plt.colorbar(orientation="horizontal",label="Electric potential (UNIT?)", norm=colors.LogNorm())        
            plt.scatter(self.x,self.y,color="red",s=5, zorder=200)

            if save==True:
                SaveFile = "Electrical Field"+ ".png" 
                plotname=os.path.join(os.getcwd(),self.folder,SaveFile)
                plt.savefig(plotname)
            plt.show()
        return(V)
    
    def lauch(self,maxtime,save=False, animation=True,animation_timing=None,phase_lag=False):
        """
        Lauch the molecular dynamic simulation of electron inside the 2DEG.
        Will stop when either we are in a steady state as definied by the minimal residue or when we reach the maximal number of iterations.

        Parameters
        ----------
        maxtime : int
            Maximal number of iteration.
        save : Bool, optional
            Do you want to save the animation. The default is False.
        animation : Bool, optional
            Do you want the see the animation, be careful this will have a major impact on the speed of the program. The default is True.
        animation_timing : int, optional
            At each frequency do you want the see the animation. The default is to see it at each iteration.
        phase_lag : Bool, optional
            Do you want to look at phase lag with your external electric field and the 2DEG. The default is False.

        Returns
        -------
        None.

        """
        while self.iterationcount<maxtime :
            self.iteration() #Let the electron evole in time
            self.boundarycondition() #force the boundary conditon on the electron
            self.diffusion() #Diffusion of the momentum into the 2DEG (scattering the the phonon/impurity)
            self.flexible() #Modifiy the min and max of the position if a electron has pass over those limits. Mostly used for bug catching purpose.
            if animation==True:
                self.show_electron(save=save) #Animation of the electron moving
            elif animation=="Timing":
                if self.iterationcount%animation_timing==0:
                    self.show_electron(save=save) # Still animation, but only at after some quantity of iterations, so not at each iteration
            
            if self.minimalresidue>self.residuex and self.minimalresidue>self.residuey and self.iterationcount>100:
                print("We are now into a steady situation.")
                break #Test if we are in a steady state as definied with the minimal residue and stop the simulation if we are in steady state
            if (self.iterationcount%100)==0:
                print("Iteration:"+str(self.iterationcount)) #just a basic counter to ensure that the simulation is not stuck somewhere
                #print(np.max((self.residuex)))
                #print(np.max(self.residuey))
            if phase_lag:
                if self.iterationcount>=1000:
                    self.show_Electrical_Field() #Show to show part of the electric field, used to show phase lag 

    def lattice(self,axis):
        """
        Show a histogram of the position of the electron for the axis chosen.
        This is a way to observe what is the lattice constants of the electrons.

        Parameters
        ----------
        axis : str
            Which axis do you want to see the histrogram, x, y or r.

        Returns
        -------
        None.

        """
        if axis=="y":
            plt.xlabel(r"$x (\mu m)$")
            plt.hist(self.x,bins=100)
        elif axis=="x":
            plt.xlabel(r"$y (\mu m)$")
            plt.hist(self.y,bins=100)
        elif axis=="r":
            plt.xlabel(r"$r (\mu m)$")
            plt.hist(np.sqrt(self.x**2+self.y**2),bins=100)
        plt.ylabel("Number of particles")
        plt.show()



def E_ext(x,y,iteration,timestep):
    omega=np.pi/100
    k=np.pi/2
    norm=np.sqrt(x**2+y**2)
    #Ex=-5e-5*x/norm/z**2
    #Ey=-5e-5*y/norm/z**2
    if iteration>=1000:
        Ex=0
        Ey=5e-5*np.cos(k*x-omega*iteration)
    else :
        Ex=0
        Ey=0
    #if iteration<750:  
    #     Ex=0 
    #else:
    #    Ex=1e-5
    #Ey=-1e-3*y**5/norm
    E=[Ex,Ey]
    
    return(np.array(E))
    #return(np.array([0,0]))

def a_ext(x,y,vx,vy,timestep,iterationcount):
    return(0)

def B(x,y,vx,vy,timestep,iteration_count):
    if iteration_count>1000:
        B=1e-7
        F_x=vy*B
        F_y=vx*B
    else :
        F_x=0
        F_y=0
    return(constants.e/constants.m_e*np.array([F_x,F_y]))



"""
Debug to do :
    -modify internal E for periodic vs reflective

"""



if __name__ == '__main__':
    global z
    folder="Picture"
    density2deg=1e3
    mean_free_path=5e-5
    relativeeps=1
    #timestep=7.5e-6
    timestep=1e-5
    minimalresidue=0.1e-5
    #minimalresidue=0
    temperature=0.2
    xmin=-1
    ymin=-1
    xmax=1
    ymax=1
    quantity=250
    initial_velocity="rest"
    timemax=10000
    z=1
    #electrongas=Simulation_Electron(folder,density2deg,mean_free_path,relativeeps,timestep,minimalresidue,temperature,E_ext,a_ext)
    #electrongas.createcloudelectron(xmin, ymin, xmax, ymax, quantity, initial_velocity)
    #print("Starting the simulation")
    #electrongas.lauch(timemax,animation=False)
    #C=electrongas.Electrical_Potential(show=True)
    folder="i"
    electrongas=Simulation_Electron(folder,density2deg,mean_free_path,relativeeps,timestep,minimalresidue,temperature,E_ext,a_ext)
    electrongas.createcloudelectron(xmin, ymin, xmax, ymax, quantity, initial_velocity)
    electrongas.lauch(5000,animation=False,phase_lag=True)
    
    
    """
    folder="SweepZ"
    V=[]
    z_list=np.linspace(1,4.5,num=5)
    electrongas=Simulation_Electron(folder,density2deg,mean_free_path,relativeeps,timestep,minimalresidue,temperature,E_ext,a_ext)
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
    def test(z,V_Estimation,electrongas):
        N=10*len(electrongas.x)
        x=np.linspace(xmin,xmax,num=N)
        y=np.linspace(ymin,ymax,num=N)
        X,Y=np.meshgrid(x,y)
        V=X*0
        for j in range(0,N):
            V[:,j]=V_Estimation((z,X[:,j],Y[:,j]))
        May_be_electron=np.abs(V)>np.mean(np.abs(V))*5
        plt.pcolor(x,y,May_be_electron)

            
            
        #Ex=V*0 
        #Ey=V*0
        #print("Start calculating the E field")
        #for j in range(0,N):
        #    Ex[:,j]=-0.5*(V[:,(j+1)%N]-V[:,(j-1)%N])/(x[1]-x[0])
        #for i in range(0,N):
        #    Ey[i,:]=-0.5*(V[(i+1)%N,:]-V[(i-1)%N,j:])/(y[1]-y[0])
        #plt.streamplot(x,y,Ex,Ey,density=2, zorder=25,color="blue")# vector field of the plot
        #plt.show()
        #return(Ex,Ey)
        
        #plt.pcolor(x,y,V)
        #plt.colorbar()
        #plt.show()
        #return(V)
    print("Start testing")
    C=test(1.93, V_Estimation, electrongas)
    
    """
    
    
    


from pickle import load
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib import pyplot as plt

lists = load("AnimationData.pkl")
parameters, timeframes = lists
"""
parameters contains all information about the simulation needed to make animation.
timeframes[0] = xpos, ypos, valE, 
where xpos, ypos give the positions of each electron, 
and valE give the values of external E field where sampled
"""

xmin, xmax, ymin, ymax, xposE = parameters
mpl.rcParams['savefig.facecolor'] = 'white'
fig, ax = plt.subplots()
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

def animate(frame_nb):
    frame = timeframes[frame_nb]
    xpos, ypos, valE = frame
    ax.clear()
    ax.set_xlabel("X [µm]")
    ax.set_ylabel("Y [µm]")
    ax.scatter(xpos, ypos,color="black",s=5, zorder=200)
    
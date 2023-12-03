from pickle import load
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import numpy as np

with open("AnimationData.pickle", 'rb') as handle:
    lists = load(handle)
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
ax.set_xlim(xmin-0.2,xmax+0.2)
ax.set_ylim(ymin-0.2,ymax+0.2)

def animate(frame_nb):
    frame = timeframes[frame_nb]
    xpos, ypos, valE = frame
    valE = np.array(valE)
    ax.clear()
    ax.set_xlabel("X [µm]")
    ax.set_ylabel("Y [µm]")
    ax.scatter(xpos, ypos,color="black",s=5, zorder=200)
    if not np.all(valE == 0):
        ax.set_title("External field: up = blue, down = red")
        plus = valE > 0
        minus = valE < 0
        plt.scatter(xposE[plus],0*xposE[plus],color="blue")
        plt.scatter(xposE[minus],0*xposE[minus],color="red")

print("Starting animation compilation")
anime = animation.FuncAnimation(fig, animate, frames=len(timeframes), interval=1)
anime.save("(To rename) Electron dynamics.gif", writer='pillow')
print("Animation saved successfully")
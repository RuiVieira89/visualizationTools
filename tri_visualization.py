
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from matplotlib import transforms

import numpy as np

import pandas as pd 


def bubbles(x, y, z, f, narea):
    
    # f - set color
    # narea - set size 
    
    def zeroToOne(self):
        return (self-min(self))/(max(self)-min(self))
    
    def doCrosses(x, y, z, f, area, cond, mark, color):
        assemble_ = [x, y, z, f, area]
        assemble = pd.concat(assemble_, axis=1, sort=False)
        assemble = assemble[assemble[area.name] < cond]    

        [xn, yn, zn, fn, arean] = assemble
        ax.scatter(assemble[xn],assemble[yn],assemble[zn],
                   marker=mark,color=color)

    
    #plots bubbles
    area = zeroToOne(narea)

    
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.set_zlabel(z.name)
    
    #fig.suptitle(['color: ',f.name, ' size: ', narea.name]) 

    pcm = ax.scatter(x, y, z, c=f, s=1000*area, cmap="RdBu_r", alpha=0.4)
    # RdBu_r
    # nipy_spectral
    cmap = fig.colorbar(pcm, ax=ax)
    cmap.set_label(f.name)
    
    # plots crosses
    cond = 0.5
    doCrosses(x, y, z, f, area, cond, 'x', 'k')    
    #doCrosses(x, y, z, f, area, cond, '+', 'r') 
    #plots lim f
    
    
def slices3D(x,y,z,f):
    
    def fTriangulation(a, b, c):
        
        x = np.array(a)
        y = np.array(b)
        z = np.array(c)

        ngridx = len(x)
        ngridy = len(y)
        npts = len(z)

        # A contour plot of irregularly spaced data coordinates
        # via interpolation on a grid.

        # Create grid values first.

        xi = np.linspace(x.min(), x.max(), ngridx)
        yi = np.linspace(y.min(), y.max(), ngridy)

        # Perform linear interpolation of the data (x,y)
        # on a grid defined by (xi,yi)
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)
        
        return [Xi, Xi, zi]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in z.unique():
        datas = pd.concat([x,y,z,f],axis=1, sort=False)
        datas = datas[z == i]
        [X,Y,funct] = fTriangulation(datas[x.name], datas[y.name], datas[f.name])
        levels = MaxNLocator(nbins=15).tick_values(funct.min(), funct.max())
        Y = np.transpose(Y)
        cntr1 = ax.contourf(X, Y, funct, zdir='z', offset=i, levels=14)
        
    ax.set_title(f.name)
    ax.set_xlabel(x.name)    
    ax.set_ylabel(y.name)
    ax.set_zlabel(z.name)
    
    ax.set_zlim(0,max(z))
    fig.colorbar(cntr1, ax=ax)  



def curve3D(x,y,z,f):
    
    axis_font ={'fontname':'Arial', 'size':'20'}
    
    def fTriangulation(a, b, c):
        
        x = np.array(a)
        y = np.array(b)
        z = np.array(c)

        ngridx = len(x)
        ngridy = len(y)
        npts = len(z)

        # A contour plot of irregularly spaced data coordinates
        # via interpolation on a grid.

        # Create grid values first.

        xi = np.linspace(x.min(), x.max(), ngridx)
        yi = np.linspace(y.min(), y.max(), ngridy)

        # Perform linear interpolation of the data (x,y)
        # on a grid defined by (xi,yi)
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)
        
        return [Xi, Yi, zi]

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')

    for i in [3,4,5]:
        datas1 = pd.concat([x,y,z,f],axis=1, sort=False)
        datas2 = pd.concat([datas1[datas1['hleg [mm]'] == 3],
                            datas1[datas1['hleg [mm]'] == 4],
                            datas1[datas1['hleg [mm]'] == 5]],
                           axis=0, sort=False)
        
        
        datas = datas1[z == i]
        [X,Y,funct] = fTriangulation(datas[x.name], datas[y.name], datas[f.name])
        levels = MaxNLocator(nbins=15).tick_values(funct.min(), funct.max()) 

           
        cntr1 = ax.plot_surface(X, Y, funct, cmap=cm.jet, alpha=0.3, 
                                linewidth=0, antialiased=False)
        ax.text(2,3, funct[4,3], 'hleg={}[mm]'.format(i),
                horizontalalignment='center')

        ax.scatter(datas[x.name], datas[y.name], datas[f.name], s=0.1, c='k')

        ax.set_xlabel(x.name, axis_font)    
        ax.set_ylabel(y.name, axis_font)
        ax.set_zlabel(f.name, axis_font)
    
    cbar = fig.colorbar(cntr1, ax=ax) 
    cbar.ax.tick_params(labelsize=axis_font['size'])

    ax.scatter(datas2.loc[datas2.idxmax()[-1]][0], 
               datas2.loc[datas2.idxmax()[-1]][1], 
               datas2.loc[datas2.idxmax()[-1]][3], 
               s=10, c='k', marker='o')
    ax.text(datas2.loc[datas2.idxmax()[-1]][0], 
               datas2.loc[datas2.idxmax()[-1]][1], 
               datas2.loc[datas2.idxmax()[-1]][3], 
               'Maximum={:.1f}'.format(datas2.loc[datas2.idxmax()[-1]][3]),
               horizontalalignment='center',
               verticalalignment='center')

def TriangulationNPlot(a, b, c):
    
    x = np.array(a)
    y = np.array(b)
    z = np.array(c)


    ngridx = len(x)
    ngridy = len(y)
    npts = len(z)

    # A contour plot of irregularly spaced data coordinates
    # via interpolation on a grid.

    # Create grid values first.

    xi = np.linspace(x.min(), x.max(), ngridx)
    yi = np.linspace(y.min(), y.max(), ngridy)


    # Perform linear interpolation of the data (x,y)
    # on a grid defined by (xi,yi)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    fig, ax = plt.subplots(1,1,figsize=(4, 3))

    ax.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")

    fig.colorbar(cntr1, ax=ax)
    ax.plot(x, y, 'ko', ms=3)
    #ax.set(xlim=(-2, 2), ylim=(-2, 2))
    ax.set_title(c.name)

    ax.set_xlabel(a.name)
    ax.set_ylabel(b.name)
    
    plt.subplots_adjust(hspace=0.5)

    
    ax = fig.add_subplot(2, 2, num, projection='3d')
    ax.plots = curve3D(x, y, z, i)





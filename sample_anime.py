import netCDF4
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
mpl.use('Agg')

filename = 'merged_history.pe000000.nc'

# for PREC
varname = 'PREC'

# # for Qhyd
varname = 'QHYD'
fct = 1000.
levels_qhyd = np.arange(0.,10.,0.5)
cmap_qhyd = 'jet'

# read
nc = netCDF4.Dataset(filename)
dim = np.ndim(nc[varname][:])
if 'dat' not in locals() :
    nt = nc.dimensions['time'].size
    nx = nc.dimensions['x'].size
    ny = nc.dimensions['y'].size
    nz = nc.dimensions['z'].size
    unit = nc[varname].units
    dat = np.empty((nt,nz,ny,nx))
    lon = np.empty((ny,nx))
    lat = np.empty((ny,nx))
    if dim == 3 : # t,y,x
        dat[:,0] = nc[varname][:]
    elif dim == 4 : # t,z,y,x
        dat = nc[varname][:]
    else :
        print('variable dimension of {:} not supported. --Abort'.format(dim))
        exit()

# make anim
ims = []
fig, ax = plt.subplots(figsize = (6, 6))
x = 0.
y = 0.101

for t in range(nt) :

    if dim == 3 :
        im = ax.plot(dat[t,0,:,0],c='k',label=varname)
        ax.set_ylim(0,0.1)
        ax.set_xlim(0,40)
        ax.set_xlabel('Y')
        ax.set_ylabel(unit)
        title = ax.text(x, y, "t = {}".format(t), fontsize=15)
        if t == 0 :
            ax.legend()
        ims.append(im + [title])

    elif dim == 4 :
        im = ax.contourf(dat[t,:,:,0] * fct,levels=levels_qhyd, cmap=cmap_qhyd)
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        title = ax.text(x, y, "t = {}".format(t), fontsize=15)
        if t == nt - 1 :
            plt.colorbar(im, extend='both')
        ims.append(im.collections + [title])

ani = animation.ArtistAnimation(fig, ims, interval=100)
ani.save('result/sample/contorl_anim_{:}.mp4'.format(varname), writer='ffmpeg')
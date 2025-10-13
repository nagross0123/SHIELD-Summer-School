# -*- coding: utf-8 -*-
"""
Script to run the rotated maps plotting routine.

Created on Wed Jun  5 09:23:30 2024

Version:  v1.2 (2024-10-22)

@author: jgasser
"""

import numpy as np
import matplotlib.pyplot as plt
import loadibex as lix
import map_rotated
# import cmcrameri.cm as cmc # used for 'batlow' colorscheme

#==============================
alf = 35  # angle between old pole and new pole
phi = -30  # longitude where the new pole is
th = 0 # end rotation towards positive azimuth
#==============================

lons, lats, data = lix.load_ibexfile_std(year=2009)
data[ np.where(data<=0) ] =np.nan

# lons, lats, data = lix.load_combifile('Hi_noSP_ram_3yr/hv60.hide-trp-flux100-hi-3-flux_2017_to_2019.csv')

fig, ax = map_rotated.plot_rotated_map(lons, lats, data, alf=alf, phi=phi, th=th,
                                       title='IBEX-Hi 1.11keV (2009)', 
                                       #cmap=cmc.batlow, grid_color= 'k',
                                       # cbar_kwargs={'location':'right','aspect':28, 'fraction':0.7},
                                       # center_meridian=True,
                                       tick_font= 30,
                                       orientation_kw= 'nose' )

plt.plot(0,0,'ok')
plt.text(.03,.02,'Nose', fontsize= 30)

plt.show()

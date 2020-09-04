import sys, os
sys.path.append('../')

import logging
import numpy as np
import matplotlib.pyplot as plt
import astropy
import pickle

from simulation.wrapper import LensingObservationWithSubhalos
from simulation.units import M_s
import paper_settings
import pyprob

paper_settings.setup()

seed=5
pyprob.seed(seed)

lo = LensingObservationWithSubhalos(f_sub=0.05,
                                    beta=-1.9,
                                    m_max_sub_div_M_hst_calib=0.01,
                                    )

plt.imshow(np.log10(lo.image_poiss_psf),
        origin="lower",
        extent=(-3.2,3.2,-3.2,3.2),
        cmap='gray'
)

plt.xlabel(r"$\theta_x$\,[as]")
plt.ylabel(r"$\theta_y$\,[as]")

plt.show()

# Substructure fraction of realization. For reasonable population parameters almost always << 1
print(lo.f_sub_realiz)

##### LATENTS #####

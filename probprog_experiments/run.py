import sys, os
sys.path.append('../')

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import astropy
import pickle
from pathlib import Path

from simulation.wrapper import LensingObservationWithSubhalos
from simulation.units import M_s
import probprog_settings
import pyprob
from pyprob.distributions import Empirical
import torch
import numpy as np

probprog_settings.setup()

choices = ['prior', 'posterior', 'ground_truth', '']
parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='results', type=str)
parser.add_argument('--num_traces', default=int(1e2), type=int)
parser.add_argument('--plot_types', nargs='+', default='', type=str,
                    choices=choices)
parser.add_argument('--gen_mode', nargs='+', default='',
                    choices=choices, type=str)

args = parser.parse_args()
base_dir = Path(args.dir)
plot_types = args.plot_types
gen_mode = args.gen_mode
num_traces = args.num_traces

directories = {d: base_dir / d
               for d in ["prior", "posterior", "ground_truth"]}

for key, new_dir in directories.items():
    if not new_dir.exists():
        new_dir.mkdir(parents=True)

#### MODEL ####

class LensingModel(pyprob.Model):

    def __init__(self):
        super(LensingModel, self).__init__()
        # TODO use ../simulation/units.py instead
        GeV = 10 ** 6
        eV = 10 ** -9 * GeV
        Kilogram = 5.6085 * 10 ** 35 * eV
        # Particle and astrophysics parameters
        self.M_s = 1.99 * 10 ** 30 * (Kilogram)

    def forward(self):
        lo = LensingObservationWithSubhalos(f_sub=0.05, beta=-1.9,
                                            m_max_sub_div_M_hst_calib=0.01,
                                            )

        # tagging
        pyprob.tag(lo.M_200_hst / self.M_s, name='host_halo_mass')
        pyprob.tag(lo.z_l, name='host_halo_redshift')
        pyprob.tag(np.array([lo.theta_x_0, lo.theta_y_0]),
                   name='host_halo_offset_xy')
        pyprob.tag(lo.n_sub_roi, name='number_subhalos')
        pyprob.tag(lo.m_subs / self.M_s, name='individual_subhalo_masses')
        pyprob.tag(np.transpose([lo.theta_xs, lo.theta_ys]),
                   name='position_of_individual_subhalos')

        pyprob.tag(lo.image_poiss_psf, name="simulated_image")
        image = np.log10(lo.image_poiss_psf)
        pyprob.tag(image, name="simulated_image_log10")

        pyprob.observe(pyprob.distributions.Normal(image.flatten(), 1),
                       name='observed_image')


#### PLOTTING FUNCITONS ####

def plot_trace(trace):

    image = trace['simulated_image_log10']

    plt.imshow(image, origin="lower", extent=(-3.2, 3.2, -3.2, 3.2),
               cmap='gray')

    plt.xlabel(r"$\theta_x$\,[as]")
    plt.ylabel(r"$\theta_y$\,[as]")

    plt.show()


def plot_distribution(dists, file_name=None, n_bins=30, num_resample=None,
                      trace=None):
    if isinstance(dists, Empirical):
        dists = [dists]

    latents_of_interest = ['host_halo_mass', 'host_halo_redshift',
                           'host_halo_offset_xy', 'number_subhalos',
                           'individual_subhalo_masses',
                           'position_of_individual_subhalos']
    marginal_dists = [{} for _ in range(len(dists))]
    pyprob.set_verbosity(0)
    for i, dist in enumerate(dists):
        if num_resample is not None:
            dist = dist.resample(num_resample)
        for lat in latents_of_interest:
            marginal_dists[i]['dist'+lat] = dist.map(lambda t: t[lat])


### Define model and other variables ##
model = LensingModel()

prior_file_name = str(directories['prior'] / 'prior.distribution')
gt_file_name = str(directories['ground_truth'] / 'gt')
posterior_file_name = str(directories['posterior'] / 'posterior.distribution')


### GENERATE DISTRIBUTIONS ###
if "prior" in gen_mode:
    print("Generating prior distribution")
    model.prior(file_name=prior_file_name,
                num_traces=num_traces)
if "ground_truth" in gen_mode:
    print("Generating ground truth")
    trace = model.sample()
    torch.save(trace, gt_file_name)
if "posterior" in gen_mode:
    print("Generating posterior distribution")
    trace = torch.load(gt_file_name)
    obs = trace['simulated_image_log10'].flatten()
    model.posterior(observe={'observed_image': obs}, num_traces=num_traces,
                    file_name=posterior_file_name)

### PLOTTING ###
if "prior" in plot_types:
    prior = Empirical(file_name=prior_file_name)
    plot_distribution(prior)
if "ground_truth" in plot_types:
    pass
if "posterior" in plot_types:
    posterior = Empirical(file_name=posterior_file_name)
    plot_distribution(posterior)

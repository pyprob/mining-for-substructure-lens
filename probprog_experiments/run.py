import sys, os
sys.path.append('../')

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy
import pickle
from pathlib import Path

from simulation.wrapper import LensingObservationWithSubhalos
import simulation.units as units
import probprog_settings
import pyprob
from pyprob.distributions import Empirical
import torch
import numpy as np
import seaborn as sns

probprog_settings.setup()

choices = ['prior', 'posterior', 'ground_truth', '']
parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='results', type=str)
parser.add_argument('--fig_dir', default='figs', type=str)
parser.add_argument('--num_traces', default=int(1e2), type=int)
parser.add_argument('--plot_types', nargs='+', default='', type=str,
                    choices=choices)
parser.add_argument('--gen_modes', nargs='+', default='',
                    choices=choices, type=str)

args = parser.parse_args()
results_dir = Path(args.results_dir)
fig_dir = Path(args.fig_dir)
plot_types = args.plot_types
gen_mode = args.gen_modes
num_traces = args.num_traces

results_directories = {d: results_dir / d
                       for d in ["prior", "posterior", "ground_truth"]}
fig_directories = {d: fig_dir / d
                   for d in ["prior", "posterior", "ground_truth"]}

for key, new_dir in results_directories.items():
    if not new_dir.exists():
        new_dir.mkdir(parents=True)

for key, new_dir in fig_directories.items():
    if not new_dir.exists():
        new_dir.mkdir(parents=True)

#### MODEL ####

class LensingModel(pyprob.Model):

    def __init__(self):
        super(LensingModel, self).__init__()
        # TODO use ../simulation/units.py instead

    def forward(self):
        lo = LensingObservationWithSubhalos(f_sub=0.05, beta=-1.9,
                                            m_max_sub_div_M_hst_calib=0.01,
                                            )

        # tagging
        pyprob.tag(lo.M_200_hst / units.M_s, name='host_halo_mass')
        pyprob.tag(lo.z_l, name='host_halo_redshift')
        pyprob.tag(np.array([lo.theta_x_0, lo.theta_y_0]),
                   name='host_halo_offset_xy')
        pyprob.tag(lo.n_sub_roi, name='number_subhalos')
        pyprob.tag(lo.m_subs / units.M_s, name='individual_subhalo_masses')
        pyprob.tag(np.transpose([lo.theta_xs, lo.theta_ys]),
                   name='position_of_individual_subhalos')

        pyprob.tag(lo.image_poiss_psf, name="simulated_image")
        image = np.log10(lo.image_poiss_psf)
        pyprob.tag(image, name="simulated_image_log10")

        pyprob.observe(pyprob.distributions.Normal(image.flatten(), 1),
                       name='observed_image')


#### PLOTTING FUNCITONS ####

def plot_trace(trace, file_name=None):

    image = trace['simulated_image_log10']

    plt.imshow(image, origin="lower", extent=(-3.2, 3.2, -3.2, 3.2),
               cmap='gray')

    plt.xlabel(r"$\theta_x$\,[as]")
    plt.ylabel(r"$\theta_y$\,[as]")

    # TODO SCATTERPLOT of the multivariate stuff each dot with a different color

    if file_name:
        print('Plotting to file: {}'.format(file_name))
        plt.savefig(file_name)
    else:
        plt.show()

def plot_distribution(dists, file_name=None, n_bins=30, num_resample=1000,
                      trace=None):
    if isinstance(dists, Empirical):
        dists = [dists]

    latents_of_interest = ['host_halo_mass', 'host_halo_redshift',
                           'host_halo_offset_xy', 'number_subhalos',
                           'individual_subhalo_masses',
                           'position_of_individual_subhalos']
    marginal_dists = [{} for _ in range(len(dists))]
    #pyprob.set_verbosity(0)
    for i, dist in enumerate(dists):
        if num_resample is not None:
            dist = dist.resample(num_resample)
        for lat in latents_of_interest:
            marginal_dists[i]['dist_'+lat] = dist.map(lambda t: t[lat])
    #pyprob.set_verbosity(2)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    t_color = 'green'
    c_color = 'red'

    # TODO deal with latents which have random length
    for i, lat in enumerate(latents_of_interest[:2] + [latents_of_interest[3]]):
        if i == 0:
            ax = fig.add_subplot(gs[0, :])
        else:
            ax = fig.add_subplot(gs[1, i-1])
        for i in range(len(dists)):
            h, bins, _ = ax.hist(marginal_dists[i]['dist_'+lat].values,
                                 bins=n_bins, alpha=0.5, density=True, zorder=2)
        ax.set_xlabel(' '.join(word.capitalize() for word in lat.split('_')))
        if trace:
            ax.vlines(trace[lat], 0, np.max(h)*1.05, linestyles='dashed',
                      zorder=3)
        ax.grid(True, zorder=1)

    # plot xy halo offset
    lat = latents_of_interest[2]
    xy_values = list(zip(*marginal_dists[i]['dist_' + lat].values))
    labels = ['$x$ offset', '$y$ offset']
    for k in range(2):
        ax = fig.add_subplot(gs[2, k])
        ax.grid(True, zorder=1)
        for i in range(len(dists)):
            label = labels[k]
            h, bins, _ = ax.hist(xy_values[k], bins=n_bins, alpha=0.5,
                                 density=True, zorder=2)
        ax.set_xlabel(label)
        if trace:
            ax.vlines(trace[lat][k], 0, np.max(h)*1.05, linestyles='dashed',
                      zorder=2)

    # TODO SCATTERPLOT of the multivariate stuff subhalo with a different color
    # each sample with alpha

    if file_name:
        print('Plotting to file: {}'.format(file_name))
        plt.savefig(file_name)
    else:
        plt.show()
    plt.close()


### Define model and other variables ##
model = LensingModel()

prior_file_name = str(results_directories['prior'] / 'prior.distribution')
gt_file_name = str(results_directories['ground_truth'] / 'gt')
posterior_file_name = str(results_directories['posterior'] / 'posterior.distribution')


### GENERATE DISTRIBUTIONS ###
if "prior" in gen_mode:
    print("Generating prior distribution")
    model.prior(file_name=prior_file_name,
                num_traces=num_traces)
if "ground_truth" in gen_mode:
    print("Generating ground truth")
    model_trace = model.sample()
    torch.save(model_trace, gt_file_name)
if "posterior" in gen_mode:
    print("Generating posterior distribution")
    model_trace = torch.load(gt_file_name)
    obs = model_trace['simulated_image_log10'].flatten()
    model.posterior(observe={'observed_image': obs}, num_traces=num_traces,
                    file_name=posterior_file_name)

### PLOTTING ###
if "prior" in plot_types:
    prior = Empirical(file_name=prior_file_name)
    plot_distribution(prior,
                      file_name=fig_directories['prior'] / 'histograms.pdf',
                      num_resample=1000)
    prior.close()
if "ground_truth" in plot_types:
    gt_trace = torch.load(gt_file_name)
    plot_trace(gt_trace,
               file_name=fig_directories['ground_truth'] / 'observed.pdf')
if "posterior" in plot_types:
    posterior = Empirical(file_name=posterior_file_name)
    obs_trace = torch.load(gt_file_name)
    plot_distribution(posterior, trace=obs_trace,
                      file_name=fig_directories['posterior'] / 'histograms.pdf')
    posterior.close()

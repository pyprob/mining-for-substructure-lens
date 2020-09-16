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
from simulation.units import *
import probprog_settings
import pyprob
from pyprob.distributions import Empirical
import torch
from torchvision.utils import make_grid
import numpy as np


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
marker_size = plt.rcParams['lines.markersize'] ** 2.

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

def plot_trace(trace, file_name=None):

    image = trace['simulated_image_log10']

    plt.imshow(image, origin="lower", extent=(-3.2, 3.2, -3.2, 3.2),
               cmap='gray')

    plt.xlabel(r"$\theta_x$\,[as]")
    plt.ylabel(r"$\theta_y$\,[as]")

    if file_name:
        print('Plotting to file: {}'.format(file_name))
        plt.savefig(file_name)
    else:
        plt.show()


def plot_distribution(dists, file_name=None, n_bins=25, num_resample=100, trace=None):
    if isinstance(dists, Empirical):
        dists = [dists]

    latents_of_interest = ['host_halo_mass', 'host_halo_redshift',
                           'host_halo_offset_xy', 'number_subhalos',
                           'individual_subhalo_masses',
                           'position_of_individual_subhalos', 'simulated_image_log10']
    marginal_dists = [{} for _ in range(len(dists))]
    # pyprob.set_verbosity(0)
    for i, dist in enumerate(dists):
        if num_resample is not None:
            dist = dist.resample(num_resample)
        for latent in latents_of_interest:
            marginal_dists[i][latent] = dist.map(lambda t: t[latent])
    # pyprob.set_verbosity(2)

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3)

    if trace:
        ax = fig.add_subplot(gs[0, 0])
        obs_image = trace['simulated_image_log10']
        ax.imshow(obs_image)
        print(obs_image.min(), obs_image.max(), obs_image.shape)
        ax.set_title('Observed image')

    ax = fig.add_subplot(gs[0, 1])
    latent = 'simulated_image_log10'
    images = np.stack(marginal_dists[0][latent].values)
    ax.imshow(images.mean(0))
    ax.set_title('Mean image: {}'.format(dists[0].name))

    if len(dists) > 1:
        ax = fig.add_subplot(gs[0, 2])
        latent = 'simulated_image_log10'
        images = np.stack(marginal_dists[1][latent].values)
        ax.imshow(images.mean(0))
        ax.set_title('Mean image: {}'.format(dists[1].name))

        ax = fig.add_subplot(gs[1, 2])
        latent = 'simulated_image_log10'
        num_samples = 16
        sample_images = [torch.from_numpy(marginal_dists[1][latent].sample()).unsqueeze(0) for _ in range(num_samples)]
        composite_image = make_grid(sample_images, nrow=4, pad_value=sample_images[0].min())[0]
        ax.imshow(composite_image.numpy())
        ax.set_title('Sample images: {}'.format(dists[1].name))

    ax = fig.add_subplot(gs[1, 0])
    latent = 'host_halo_mass'
    bins = np.histogram(np.hstack([marginal_dists[i][latent].values for i in range(len(dists))]), bins=n_bins)[1]
    for i in range(len(dists)):
        ax.hist(marginal_dists[i][latent].values, bins=bins, alpha=0.5, density=True, label=dists[i].name, color=colors[i])
    ax.set_xlabel('Host halo mass')
    ax.legend()
    if trace:
        ax.axvline(trace[latent], linestyle='dashed', color='black')

    ax = fig.add_subplot(gs[1, 1])
    latent = 'host_halo_redshift'
    bins = np.histogram(np.hstack([marginal_dists[i][latent].values for i in range(len(dists))]), bins=n_bins)[1]
    for i in range(len(dists)):
        ax.hist(marginal_dists[i][latent].values, bins=bins, alpha=0.5, density=True, color=colors[i])
    ax.set_xlabel('Host halo redshift')
    if trace:
        ax.axvline(trace[latent], linestyle='dashed', color='black')

    ax = fig.add_subplot(gs[2, 0])
    latent = 'number_subhalos'
    bins = np.histogram(np.hstack([marginal_dists[i][latent].values for i in range(len(dists))]), bins=n_bins)[1]
    for i in range(len(dists)):
        ax.hist(marginal_dists[i][latent].values, bins=bins, alpha=0.5, density=True, color=colors[i])
    ax.set_xlabel('Number of subhalos')
    if trace:
        ax.axvline(trace[latent], linestyle='dashed', color='black')

    ax = fig.add_subplot(gs[2, 1])
    for i in range(len(dists)):
        dist_positions = marginal_dists[i]['position_of_individual_subhalos'].values
        dist_masses = marginal_dists[i]['individual_subhalo_masses'].values
        for positions, masses in zip(dist_positions, dist_masses):
            marker_sizes = (np.log10(masses) - 7) * (11-7) * marker_size
            ax.scatter(positions[:, 0], positions[:, 1], alpha=0.1, color=colors[i], s=marker_sizes)
    ax.set_xlabel('Subhalo position x')
    ax.set_ylabel('Subhalo position y')
    if trace:
        positions = trace['position_of_individual_subhalos']
        masses = trace['individual_subhalo_masses']
        marker_sizes = (np.log10(masses) - 7) * (11-7) * marker_size
        ax.scatter(positions[:, 0], positions[:, 1], alpha=0.75, color='black', s=marker_sizes)

    ax = fig.add_subplot(gs[2, 2])
    for i in range(len(dists)):
        positions = marginal_dists[i]['host_halo_offset_xy'].values_numpy()
        masses = marginal_dists[i]['host_halo_mass'].values_numpy()
        marker_sizes = (np.log10(masses) - 12) * marker_size
        ax.scatter(positions[:, 0], positions[:, 1], alpha=0.2, color=colors[i], s=marker_sizes)
    ax.set_xlabel('Host halo offset x')
    ax.set_ylabel('Host halo offset y')
    if trace:
        position = trace['host_halo_offset_xy']
        mass = trace['host_halo_mass']
        ms = (np.log10(mass) - 12) * marker_size
        ax.scatter(position[0], position[1], color='black', s=ms)
        ax.axvline(position[0], linestyle='dashed', color='black')
        ax.axhline(position[1], linestyle='dashed', color='black')

    plt.tight_layout()
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
prior = None
if "prior" in plot_types:
    print('Plotting prior')
    prior = Empirical(file_name=prior_file_name)
    plot_distribution(prior,
                      file_name=fig_directories['prior'] / 'histograms.pdf',
                      num_resample=None)
if "ground_truth" in plot_types:
    print('Plotting ground truth')
    gt_trace = torch.load(gt_file_name)
    plot_trace(gt_trace,
               file_name=fig_directories['ground_truth'] / 'observed.pdf')
if "posterior" in plot_types:
    print('Plotting posterior')
    posterior = Empirical(file_name=posterior_file_name)
    gt_trace = torch.load(gt_file_name)

    if prior is not None:
        dists = [prior, posterior]
    else:
        dists = [posterior]

    plot_distribution(dists, trace=gt_trace,
                      file_name=fig_directories['posterior'] / 'histograms.pdf')
    posterior.close()

if prior is not None:
    prior.close()

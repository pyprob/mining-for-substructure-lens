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
from torchvision.utils import make_grid
import numpy as np
import seaborn as sns


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
marker_size = plt.rcParams['lines.markersize'] ** 2.

probprog_settings.setup()

choices_plot = ['prior', 'posterior_IS', 'posterior_IC', 'ground_truth']
choices_gen = ['prior', 'posterior_IS', 'ground_truth', 'posterior_IC', 'data']
parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='results', type=str)
parser.add_argument('--fig_dir', default='figs', type=str)
parser.add_argument('--num_traces', default=int(1e2), type=int)
parser.add_argument('--plot', nargs='+', default='', type=str,
                    choices=choices_plot)
parser.add_argument('--gen', nargs='+', default='', choices=choices_gen,
                    type=str)
parser.add_argument('--train_ic', action="store_true")

args = parser.parse_args()
results_dir = Path(args.results_dir)
fig_dir = Path(args.fig_dir)
plot_mode = args.plot
gen_mode = args.gen
num_traces = args.num_traces
train_ic = args.train_ic

results_directories = {d: results_dir / d
                       for d in choices_gen + ['models']}
fig_directories = {d: fig_dir / d
                   for d in choices_plot}

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


def plot_distribution(dists, file_name=None, n_bins=25, num_resample=1000,
                      trace=None):
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
            ax.scatter(positions[:, 0], positions[:, 1], alpha=0.05, color=colors[i], s=marker_sizes)
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
posterior_names = {"posterior_IS": str(results_directories['posterior_IS'] /
                                       'posterior_IS.distribution'),
                   "posterior_IC": str(results_directories['posterior_IC'] /
                                       'posterior_IC.distribution')}
### Generate data ###
if "data" in gen_mode:
    print("Generating data")
    model.save_dataset(results_directories['data'],
                       num_traces, num_traces)

### Train Inference Network ###
if train_ic:
    obs_embeds = {'observed_image': {'dim': 200}}
    files_generator = results_directories['data'].glob('pyprob_traces*')

    def is_empty(generator):
        try:
            next(generator)
            return False
        except StopIteration:
            return True

    if not is_empty(files_generator):
        dataset_dir = str(results_directories['data'])
    else:
        dataset_dir = None
    save_path = results_directories['models'] / "ic"
    # TODO validation stuff
    model.learn_inference_network(num_traces=num_traces,
                                  dataset_dir=dataset_dir,
                                  observe_embeddings=obs_embeds,
                                  save_file_name_prefix=str(save_path),
                                  inference_network=pyprob.InferenceNetwork.LSTM)

### GENERATE DISTRIBUTIONS ###

if "prior" in gen_mode:
    print("Generating prior distribution")
    model.prior(file_name=prior_file_name,
                num_traces=num_traces)
if "ground_truth" in gen_mode:
    print("Generating ground truth")
    model_trace = model.sample()
    torch.save(model_trace, gt_file_name)
if sum("posterior" in mode for mode in gen_mode) > 0:
    posteriors = [mode for mode in gen_mode if "posterior" in mode]
    for pos in posteriors:
        print(f"Generating {pos} distribution")
        model_trace = torch.load(gt_file_name)
        obs = model_trace['observed_image'].flatten()
        model.posterior(observe={'observed_image': obs}, num_traces=num_traces,
                        file_name=posterior_names[pos])

### PLOTTING ###
prior = None
posterior = None
if "prior" in plot_mode:
    print('Plotting prior')
    prior = Empirical(file_name=prior_file_name)
    plot_distribution(prior,
                      file_name=fig_directories['prior'] / 'histograms.pdf')
if "ground_truth" in plot_mode:
    print('Plotting ground truth')
    gt_trace = torch.load(gt_file_name)
    plot_trace(gt_trace,
               file_name=fig_directories['ground_truth'] / 'observed.pdf')
if sum("posterior" in mode for mode in gen_mode) > 0:
    posteriors = [mode for mode in gen_mode if "posterior" in mode]
    for pos in posteriors:
        print(f'Plotting {pos}')
        posterior = Empirical(file_name=posterior_names[pos])
        gt_trace = torch.load(gt_file_name)

        if prior is not None:
            dists = [prior, posterior]
        else:
            dists = [posterior]

        plot_distribution(dists, trace=gt_trace,
                          file_name=fig_directories[pos]
                          / 'histograms.pdf')
        posterior.close()


# Close Empiricals
if prior is not None:
    prior.close()
if posterior is not None:
    posterior.close()

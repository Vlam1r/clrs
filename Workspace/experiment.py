import subprocess

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def run_experiment(name: str, path: str, global_paths_drawn=64, traj_drawn=64):

  data_dump = np.load(path + '.npz')
  data = data_dump['trajs']
  score = data_dump['score']

  samples, mp_steps, dim = data.shape

  trajwise_pca = PCA()
  trajwise = trajwise_pca.fit_transform(data.reshape((samples, mp_steps * dim)))
  assert trajwise.shape == (samples, min(samples, mp_steps * dim))

  # Reorder data according to trajwise pca x coordinate
  # Needed for sampling lines and maybe sequential experiments
  order = np.argsort(trajwise[:, 0])
  data = data[order, :, :]
  trajwise = trajwise[order, :]

  stepwise_global_pca = PCA()
  stepwise_global = stepwise_global_pca.fit_transform(data.reshape((samples * mp_steps, dim)))
  assert stepwise_global.shape == (samples * mp_steps, min(samples * mp_steps, dim))

  stepwise_local = np.array([PCA(n_components=3).fit_transform(data[:, i, :]) for i in range
  (mp_steps)])
  assert stepwise_local.shape == (mp_steps, samples, 3)

  def get_pca_evr(pca, d):
    return 100 * pca.explained_variance_ratio_[:d].sum()

  PIXEL_S = (72. / 300) ** 2
  common_scatter_args = lambda s: dict(marker='o', s=s, edgecolor='none')
  common_plot_args = dict(color='maroon', alpha=0.6, lw=0.1)

  # Plot all hidden dims as independent datapoints
  fig = plt.figure(figsize=(12, 6))
  ax = [fig.add_subplot(121), fig.add_subplot(122, projection='3d')]

  for i in range(mp_steps):
    ax[0].scatter(stepwise_global[i::mp_steps, 0],
                  stepwise_global[i::mp_steps, 1],
                  label=f'mp step {i + 1}', **common_scatter_args(PIXEL_S))
    ax[1].scatter3D(stepwise_global[i::mp_steps, 0],
                    stepwise_global[i::mp_steps, 1],
                    stepwise_global[i::mp_steps, 2],
                    label=f'mp step {i + 1}', **common_scatter_args(PIXEL_S))

  if global_paths_drawn > 0:
    for i in np.arange(0, samples, step=int(samples / global_paths_drawn)):
      assert stepwise_global[i * mp_steps:(i + 1) * mp_steps, :].shape == (mp_steps, dim)
      ax[0].plot(stepwise_global[i * mp_steps:(i + 1) * mp_steps, 0],
                 stepwise_global[i * mp_steps:(i + 1) * mp_steps, 1],
                 **common_plot_args)
      ax[1].plot3D(stepwise_global[i * mp_steps:(i + 1) * mp_steps, 0],
                   stepwise_global[i * mp_steps:(i + 1) * mp_steps, 1],
                   stepwise_global[i * mp_steps:(i + 1) * mp_steps, 2],
                   **common_plot_args)

  fig.suptitle(f"Pointwise PCA (shape = samples*mp_steps, dim)\n "
               f"{get_pca_evr(stepwise_global_pca, 3):.2f}% explained")

  fig.tight_layout()
  fig.savefig(f"plots/{name}_stepwise_global.png", dpi=300)

  # Plot trajectories
  fig = plt.figure(figsize=(12, 6))
  ax = [fig.add_subplot(121), fig.add_subplot(122, projection='3d')]

  ax[0].scatter(trajwise[:, 0], trajwise[:, 1], c=score, vmin=0, vmax=1, cmap='viridis_r',
                **common_scatter_args(16 * PIXEL_S))
  ax[1].scatter3D(trajwise[:, 0], trajwise[:, 1], trajwise[:, 2], c=score, vmin=0, vmax=1, cmap='viridis_r',
                  **common_scatter_args(16 * PIXEL_S))

  fig.suptitle(f"Trajectory-wise PCA (shape = samples, mp_steps*dim)\n "
               f"{get_pca_evr(trajwise_pca, 3):.2f}% explained")
  fig.tight_layout()
  fig.savefig(f"plots/{name}_trajwise.png", dpi=300)

  SCALE_FACTOR = 2 + mp_steps
  trajs_PCA_scale_factor = 1 / np.max(np.abs(stepwise_local[:, :, 2]), axis=-1) / SCALE_FACTOR

  fig, ax = plt.subplots(figsize=(15, 6), subplot_kw=dict(projection=f'3d'))
  ax.view_init(azim=-75, elev=15)
  ax.set_box_aspect(aspect=(2.5, 1, 1))
  for i in range(mp_steps):
    ax.scatter([i] * samples + stepwise_local[i, :, 2] * trajs_PCA_scale_factor[i],
               stepwise_local[i, :, 0], stepwise_local[i, :, 1], **common_scatter_args(PIXEL_S))
  if traj_drawn > 0:
    for i in np.arange(0, samples, step=int(samples / traj_drawn)):
      ax.plot(np.arange(mp_steps) + stepwise_local[:, i, 2] * trajs_PCA_scale_factor,
              stepwise_local[:, i, 0], stepwise_local[:, i, 1],
              **common_plot_args)

  fig.tight_layout()
  fig.savefig(f"plots/{name}_stepwise_local.png", dpi=300)

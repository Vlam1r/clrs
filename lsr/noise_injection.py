from enum import Enum
from sklearn.decomposition import PCA
from typing import Tuple
from functools import partial

import jax.numpy as jnp
import jax
import numpy as np


class NoiseInjectionStrategy(Enum):
  Noisefree = 0
  Uniform = 1
  Directional = 2
  Project = 3
  Discard = 4
  Corrupt = 5


def load_noise_vectors(mode: NoiseInjectionStrategy):
  if mode in [NoiseInjectionStrategy.Discard, NoiseInjectionStrategy.Directional, NoiseInjectionStrategy.Project]:
      data = np.load('Workspace/noise_dirs.npz')
      mus = jnp.array(data['mus'])
      evs = data['eigens']  # * data['values'][..., np.newaxis]
      return {'mus': mus, 'evs': evs}
  else:
      return None


@jax.jit
def project_on_k(v: jnp.ndarray, dirs: jnp.ndarray):
  @partial(jnp.vectorize, signature='(n),(n)->(n)')
  def project(x, y):
    return y * jnp.dot(x, y) / jnp.dot(y, y)

  return np.sum(project(v, dirs), axis=0)


@jax.jit
def select_optimal_direction_from_reference_directions(vector: jnp.ndarray,
                                                       refs: jnp.ndarray,
                                                       rng: jax.random.PRNGKeyArray,
                                                       idx: int):
  mus = refs['mus']
  evs = refs['evs']

  # idy = jax.random.choice(rng, evs.shape[1])
  # return evs[idx, idy, ...]

  # Renormalized mean
  dirs = jnp.mean(evs, axis=0)
  dirs = dirs / jnp.linalg.norm(dirs)
  return dirs[:, 0, :]

  #L2 closest
  # idy = jnp.argmin(jnp.linalg.norm(mus-vector, axis=1))
  # return evs[idx, idy, ...]

  # Random direction from references
  # idy = jax.random.choice(rng, evs.shape[1])
  # return evs[idx, idy, ...]


@partial(jax.jit, static_argnames=['mode'])
@partial(jnp.vectorize, signature='(n),()->(n)', excluded=[1, 2, 3, 4])
def inject_noise(vector: jnp.ndarray,
                 refs: jnp.ndarray,
                 mode: NoiseInjectionStrategy,
                 rng: jax.random.PRNGKeyArray,
                 idx: int,
                 length: jnp.ndarray):

  if mode == NoiseInjectionStrategy.Noisefree:
      return vector

  elif mode == NoiseInjectionStrategy.Uniform:
      # Control Experiment to Directional
      # This should be equivalent to adding noise along a randomly chosen direction with magnitude in N(0, 1)

      noise = jax.random.normal(rng, shape=vector.shape) / jnp.sqrt(vector.shape[0])
      return vector + noise

  elif mode == NoiseInjectionStrategy.Directional:
      dirs = select_optimal_direction_from_reference_directions(vector, refs, rng, idx)
      noise = jax.random.normal(rng, shape=[dirs.shape[0]])
      noise = noise @ dirs
      return vector + noise

  elif mode == NoiseInjectionStrategy.Project:
      return project_on_k(vector, select_optimal_direction_from_reference_directions(vector, refs, rng, idx))

  elif mode == NoiseInjectionStrategy.Discard:
      return vector - project_on_k(vector, select_optimal_direction_from_reference_directions(vector, refs, rng, idx))
  elif mode == NoiseInjectionStrategy.Corrupt:
      # Add noise after 10 message passing rounds (and after convergence is achieved)
      # to test how robust the network is to noise
      flag = jnp.ones_like(vector) * (1 - (idx == length - 1))
      noise = jax.random.normal(rng, shape=vector.shape)
      return vector * flag + noise * (1 - flag)

  else:
      raise ValueError


###############
# GENERATE
###############


@partial(np.vectorize, signature='(n,m)->(m),(k,m),(k)', excluded=['dim'])
def process_points(points: np.ndarray, dim=2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  pca = PCA()
  pca.fit(points)
  return np.mean(points, axis=0), pca.components_[:dim, :], pca.explained_variance_[:dim]


def find_direction(data, split):
  samples, mp_steps, dim = data.shape
  data = data.transpose(1, 0, 2).reshape((mp_steps * samples // split, split, dim))
  mus, eigens, values = process_points(data)
  return mus, eigens, values


def main(path, split):
  data_dump = np.load(path)
  data = data_dump['trajs']
  samples, mp_steps, dim = data.shape
  if split < 0:
    split = data.shape[0] // (-split)
  mus, eigens, values = find_direction(data, split)
  eigens = eigens.reshape((mp_steps, samples // split, -1, dim))
  np.savez('noise_dirs.npz', mus=mus, eigens=eigens, values=values)


if __name__ == '__main__':
  main(path='trajectories/random_johnson_512.npz', split=512)

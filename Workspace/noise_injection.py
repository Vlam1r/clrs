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


def load_noise_vectors(mode: NoiseInjectionStrategy):
  if mode in {NoiseInjectionStrategy.Directional,
              NoiseInjectionStrategy.Project,
              NoiseInjectionStrategy.Discard}:
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
def select_optimal_direction_from_reference_directions(vector: jnp.ndarray, refs, rng: jax.random.PRNGKeyArray):
  mus = refs['mus']
  evs = refs['evs']

  # Renormalized mean
  # dirs = jnp.mean(evs, axis=0)
  # dirs = dirs / jnp.linalg.norm(dirs)

  # L2 closest
  # idx = jnp.argmin(jnp.linalg.norm(mus-vector, axis=1))
  # return evs[idx]

  # Random direction from references
  # idx = jax.random.choice(rng, evs.shape[0])
  # return evs[idx]

  # Random direction
  return (jax.random.normal(rng, shape=vector.shape) / jnp.sqrt(vector.shape[0]))[np.newaxis, ...]


@partial(jax.jit, static_argnames=['mode'])
@partial(jnp.vectorize, signature='(n)->(n)', excluded=[1, 2, 3, 4])
def inject_noise(vector: jnp.ndarray, refs, mode: NoiseInjectionStrategy, rng: jax.random.PRNGKeyArray, idx: int):
  if mode == NoiseInjectionStrategy.Noisefree:
    return vector
  elif mode == NoiseInjectionStrategy.Uniform:
    # Control Experiment to Directional
    # This should be equivalent to adding noise along a randomly chosen direction with magnitude in N(0, 1)
    noise = jax.random.normal(rng, shape=vector.shape) / jnp.sqrt(vector.shape[0])
    return vector + noise
  elif mode == NoiseInjectionStrategy.Directional:
    dirs = select_optimal_direction_from_reference_directions(vector, refs, rng)
    noise = jax.random.normal(rng, shape=[dirs.shape[0]])
    noise = noise @ dirs
    return vector + noise
  elif mode == NoiseInjectionStrategy.Project:
    return project_on_k(vector, select_optimal_direction_from_reference_directions(vector, refs, rng))
  elif mode == NoiseInjectionStrategy.Discard:
    return vector - project_on_k(vector, select_optimal_direction_from_reference_directions(vector, refs, rng))
  else:
    raise ValueError

###############
# GENERATE
###############


def find_direction(path, split):
    data_dump = np.load(path + '.npz')
    data = data_dump['trajs']
    samples, mp_steps, dim = data.shape

    if split < 0:
        split = data.shape[0] // (-split)

    @partial(np.vectorize, signature='(n,m)->(m),(k,m),(k)', excluded=['dim'])
    def process_points(points: np.ndarray, dim=5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pca = PCA()
        pca.fit(points)
        return np.mean(points, axis=0), pca.components_[:dim, :], pca.explained_variance_[:dim]

    # print(data.shape)
    data = data.transpose(1, 0, 2).reshape((mp_steps * samples // split, split, dim))
    # print(data.shape)
    mus, eigens, values = process_points(data)
    print(mus.shape)
    print(eigens.shape)
    print(values.shape)
    eigens = eigens.reshape((mp_steps, samples // split, -1))
    values = values.reshape((mp_steps, -1))
    # print(mus.shape)
    # print(eigens.shape)
    np.savez('noise_dirs.npz', mus=mus, eigens=eigens, values=values)


if __name__ == '__main__':
  find_direction(path='random_johnson_n64_noisefree', split=-64)

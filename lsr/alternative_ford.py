from typing import Tuple

import numpy as np

def bellman_ford(A: np.ndarray, s: int, len: int) -> Tuple[np.ndarray, np.ndarray]:
  """Bellman-Ford's single-source shortest path (Bellman, 1958)."""

  DISCONNECTED = 0

  pis = []

  d = np.zeros(A.shape[0])
  pi = np.arange(A.shape[0])
  msk = np.zeros(A.shape[0])
  d[s] = DISCONNECTED
  msk[s] = 1
  for i in range(len):
    prev_d = np.copy(d)
    prev_msk = np.copy(msk)

    pis.append(np.copy(pi))

    for u in range(A.shape[0]):
      for v in range(A.shape[0]):
        if prev_msk[u] == 1 and A[u, v] != DISCONNECTED:
          if msk[v] == 0 or prev_d[u] + A[u, v] < d[v]:
            d[v] = prev_d[u] + A[u, v]
            pi[v] = u
          msk[v] = 1
    # if np.all(d == prev_d):
    #   break

  return np.array(pis), np.array(d)


def decay_ford(A: np.ndarray, s: int, len: int) -> np.ndarray:

  for i in range(A.shape[0]):
    A[i, i] = 0
  A[s, s] = 0

  DISCONNECTED = 0

  pis = []

  d = np.zeros(A.shape[0])
  pi = np.arange(A.shape[0])
  msk = np.zeros(A.shape[0])
  d[s] = DISCONNECTED
  msk[s] = 1
  for i in range(len):
    prev_d = np.copy(d)
    prev_msk = np.copy(msk)

    pis.append(np.copy(pi))

    for u in range(A.shape[0]):
      for v in range(A.shape[0]):
        if prev_msk[u] == 1 and A[u, v] != DISCONNECTED:
          if msk[v] == 0 or prev_d[u] + A[u, v] < d[v] * 0.9:
            d[v] = (prev_d[u] + A[u, v])
            pi[v] = u
          msk[v] = 1
    # if np.all(d == prev_d):
    #   break

  return np.array(pis)

def noisy_ford(A: np.ndarray, s: int, len: int) -> np.ndarray:

  for i in range(A.shape[0]):
    A[i, i] = 0
  A[s, s] = 0

  DISCONNECTED = 0

  pis = []

  d = np.zeros(A.shape[0])
  pi = np.arange(A.shape[0])
  msk = np.zeros(A.shape[0])
  d[s] = DISCONNECTED
  msk[s] = 1
  for i in range(len):
    prev_d = np.copy(d)
    prev_msk = np.copy(msk)

    pis.append(np.copy(pi))

    for u in range(A.shape[0]):
      for v in range(A.shape[0]):
        if prev_msk[u] == 1 and A[u, v] != DISCONNECTED:
          if msk[v] == 0 or prev_d[u] + A[u, v] < d[v] + np.random.uniform(-0.05, 0.05):
            d[v] = (prev_d[u] + A[u, v])
            pi[v] = u
          msk[v] = 1
    # if np.all(d == prev_d):
    #   break

  return np.array(pis)



def greedy_ford(A: np.ndarray, s: int, len: int) -> np.ndarray:

  for i in range(A.shape[0]):
    A[i, i] = 0
  A[s, s] = 0

  DISCONNECTED = 0

  pis = []

  d = np.zeros(A.shape[0])
  pi = np.arange(A.shape[0])
  msk = np.zeros(A.shape[0])
  d[s] = DISCONNECTED
  msk[s] = 1
  for i in range(len):
    prev_d = np.copy(d)
    prev_msk = np.copy(msk)

    pis.append(np.copy(pi))

    for u in range(A.shape[0]):
      for v in range(A.shape[0]):
        if prev_msk[u] == 1 and A[u, v] != DISCONNECTED:
          if msk[v] == 0 or A[u, v] < A[pi[v], v]:  # prev_d[u] + A[u, v] < d[v]:
            d[v] = prev_d[u] + A[u, v]
            pi[v] = u
          msk[v] = 1
    # if np.all(d == prev_d):
    #   break

  return np.array(pis)


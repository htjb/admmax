"""Example use case for the qp solver."""

import time
from itertools import product

import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp

from admmax.admm import admm_qp
from maxsmooth.derivatives import derivative_prefactors
from maxsmooth.models import *

jax.config.update("jax_enable_x64", True)

function = difference_polynomial
basis_function = difference_polynomial_basis

key = jax.random.PRNGKey(0)
x = jnp.linspace(50, 150, 100)
y = 5e6 * x ** (-2.5) + 0.01 * jax.random.normal(key, x.shape)
N = 12
pivot_point = len(x) // 2

start = time.time()

x_pivot = x[pivot_point]
y_pivot = y[pivot_point]
# needs some dummy parameters to make basis
basis_function = jax.vmap(basis_function, in_axes=(0, None, None, None))
basis = basis_function(x, x_pivot, y_pivot, jnp.ones(N))
Q = jnp.dot(basis.T, basis)

c = -jnp.dot(basis.T, y)
G = derivative_prefactors(function, x, x_pivot, y_pivot, jnp.ones(N), N)[2:]
G = jnp.array(G)

print(jnp.max(Q))

all_signs = jnp.array(list(product((-1.0, 1.0), repeat=len(G))))

admm_vmap = jax.vmap(admm_qp, in_axes=(0, None, None, None))
p, iters, converged, s, u, viol = admm_vmap(all_signs, Q, c, G)
print(converged)
print(viol)
print(f"ADMM QP took {time.time() - start:.3f} seconds")


vmapped_fit = jax.vmap(function, in_axes=(0, None, None, None))
objective_vals = []
fits = []
passed_signs = []
for i in range(len(all_signs)):
    passed_signs.append(all_signs[i])
    fit = vmapped_fit(x, x[pivot_point], y[pivot_point], p[i])
    objective = jnp.sum((y - fit) ** 2)
    objective_vals.append(objective)
    fits.append(fit)
objective_vals = jnp.array(objective_vals)
fits = jnp.array(fits)
ob = jnp.array(
    [objective_vals[i] for i in range(len(objective_vals)) if converged[i]]
)
best_idx = jnp.argmin(ob)

fig, axes = plt.subplots(1, 2, figsize=(5, 5))
axes[0].plot(x, fits[best_idx], label=f"Sign: {all_signs[best_idx]}")
axes[0].scatter(x, y, s=5, color="black", label="Data")
axes[0].legend()
axes[1].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("ADMM QP fits with different derivative sign constraints")
axes[1].plot(
    x,
    y - fits[best_idx],
    "o",
    label="Residuals",
    markersize=3,
)
plt.savefig("admm_qp_example.png", dpi=300)
plt.close()

plt.scatter(
    jnp.arange(len(objective_vals)),
    objective_vals,
    c=["blue" if converged[i] else "red" for i in range(len(objective_vals))],
    label="Objective Values",
)
plt.yscale("log")
plt.xlabel("Index")
plt.ylabel("Objective Value")
plt.title("Objective values for converged (blue) and non-converged (red) fits")
plt.savefig("admm_qp_objective_values.png", dpi=300)
plt.close()

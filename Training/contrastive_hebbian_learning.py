"""Contrastive Hebbian learning (daydreaming) training for Model A.

Functions here train the interaction matrix `chi` (and chemical potential
offsets `miu`) for the Model A dynamics using a contrastive Hebbian /
daydreaming scheme with an AdamW optimizer, plus helpers for evaluating
recall/retrieval performance of a trained model.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
import optax
from tqdm.auto import tqdm

from Dynamics.Model_A import forward_sim_x_ssolvent_clamp, x_to_phi


def partition_memories_softmax(memories, temperature=0.5):
    """Apply a softmax with the given temperature to each memory row."""
    exp_memories = np.exp(memories / temperature)
    partitioned = exp_memories / np.sum(exp_memories, axis=1, keepdims=True)
    return partitioned


def daydreaming_contrastive_hebbian_learning_lagrange_JAX_fast_sample_adamw(
    target_memories, chi_initial, miu_initial,
    V=1.0, gamma_dynamics=1.0, dt_dynamics=1e-2, n_steps_dynamics=30000,
    gamma_learning=10.0, n_epochs=200, clamped=None,
    n_sample=None, sample_random_each_epoch=False,
    p_boundary=0.3, t_end=300.0, width=0.02,
    weight_decay=1e-5,
    print_energy=True, key_seed=42
):
    target_memories = None if target_memories is None else jnp.array(target_memories)
    chi_initial = jnp.array(chi_initial)
    miu_initial = jnp.array(miu_initial)

    @jit
    def flory_huggins_free_energy_jax(phi, chi, miu, V):
        phi = jnp.clip(phi, 1e-10, 1.0)
        entropy = jnp.sum(phi * jnp.log(phi))
        interaction = 0.5 * phi @ chi @ phi
        potential = jnp.sum(miu * phi)
        return V * (entropy + interaction + potential)

    # ---------------- Single Memory Update ----------------
    def update_chi_single(key, phi, chi, miu, V, gamma_dynamics, dt_dynamics,
                           n_steps_dynamics, clamped=None):
        N = phi.shape[0]
        phi_ini = phi
        # Positive phase
        phi_plus = phi

        # Negative phase
        phi_minus_diffrax = forward_sim_x_ssolvent_clamp(
            phi0=phi_plus,
            chi=chi,
            mu=miu,
            clamp=0,  # fully free for negative phase
            t_end=t_end,
            dt=dt_dynamics,
            samples=10,
            max_steps=n_steps_dynamics,
        )
        phi_minus = x_to_phi(phi_minus_diffrax.ys[-1])

        # Energy
        F_plus = flory_huggins_free_energy_jax(phi_plus, chi, miu, V)
        F_minus = flory_huggins_free_energy_jax(phi_minus, chi, miu, V)
        energy_diff = F_plus - F_minus

        def compute_true_error_swap(phi_ini, phi_minus, N):
            pred = phi_ini[2] > phi_ini[3]

            i_hi = jnp.where(pred, 2, 3)
            i_lo = jnp.where(pred, 3, 2)

            return (
                jnp.log(1 + N * jnp.maximum(1.1 / N - phi_minus[i_hi], 0.0)) +
                jnp.log(1 + N * jnp.minimum(phi_minus[i_lo] - 0.25 / N, 0.0))
            )

        # Gradients (for AdaGrad)
        true_error = compute_true_error_swap(phi_ini, phi_minus, N)
        grad_scaling = 1.0
        grad_chi = (jnp.outer(phi_plus, phi_plus) - jnp.outer(phi_minus, phi_minus)) * grad_scaling
        # miu is not updated in this version, but we compute the grad for potential future use
        grad_miu = (phi_plus - phi_minus) * grad_scaling * 0.1

        # Zero out fixed elements
        grad_chi = grad_chi.at[jnp.diag_indices(grad_chi.shape[0])].set(0.0)

        return grad_chi, grad_miu, energy_diff, true_error

    # ---------------- Random-memory generator ----------------
    def make_random_memories(key, sample_size, N):
        keys = jax.random.split(key, sample_size + 1)
        key_next = keys[0]
        mem_keys = keys[1:]

        def make_one(k):
            k1, k2, k3, k4 = jax.random.split(k, 4)
            m1 = jax.random.uniform(k3, ()) < p_boundary
            m2 = jax.random.uniform(k4, ()) < 0.5

            # sample uniform
            inp0_u = jax.random.uniform(k1, shape=(), minval=0.0, maxval=0.25)
            inp1_u = jax.random.uniform(k2, shape=(), minval=0.0, maxval=0.25)
            # sample near boundary
            inp0_b1 = jax.random.uniform(k1, shape=(), minval=0.125 - width, maxval=0.125 + width)
            inp1_b1 = jax.random.uniform(k2, shape=(), minval=0.125 - width, maxval=0.25)

            inp0_b = jnp.where(m2, inp0_b1, inp1_b1)
            inp1_b = jnp.where(m2, inp1_b1, inp0_b1)

            inp0 = jnp.where(m1, inp0_u, inp0_b)
            inp1 = jnp.where(m1, inp1_u, inp1_b)

            out = jnp.zeros(N)
            out = out.at[0].set(inp0).at[1].set(inp1)

            cond = (inp0 > 0.125) & (inp1 > 0.125)
            hi, lo = 1.1 / N, 0.25 / N
            val2 = jnp.where(cond, hi, lo)
            val3 = jnp.where(cond, lo, hi)
            out = out.at[2].set(val2).at[3].set(val3)

            sum_fixed = jnp.sum(out[:4])
            sum_rem = jnp.sum(out[4:])
            out = out.at[4:].set(out[4:] / sum_rem * (1 - sum_fixed))
            return out

        mems = jax.vmap(make_one)(mem_keys)
        return key_next, mems

    # ---------------- AdamW optimizer ----------------
    params = (chi_initial, miu_initial)
    optimizer = optax.adamw(learning_rate=gamma_learning, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    # ---------------- Batched Update ----------------
    def batched_update(key, chi, miu, opt_state, clamped):
        sample_size = n_sample if n_sample is not None else (target_memories.shape[0] if target_memories is not None else 1)
        N = chi.shape[0]
        if sample_random_each_epoch or (target_memories is None):
            key, sampled_memories = make_random_memories(key, sample_size, N)
        else:
            num_memories = target_memories.shape[0]
            key, subkey = jax.random.split(key)
            replace = sample_size > num_memories
            indices = jax.random.choice(subkey, num_memories, (sample_size,), replace=replace)
            sampled_memories = target_memories[indices]

        mem_keys = jax.random.split(key, sample_size + 1)
        key = mem_keys[0]
        mem_keys = mem_keys[1:]

        grad_chi_list, grad_miu_list, energy_diffs, true_errors = jax.vmap(
            lambda k, mem: update_chi_single(k, mem, chi, miu, V, gamma_dynamics, dt_dynamics, n_steps_dynamics, clamped)
        )(mem_keys, sampled_memories)

        grad_chi = jnp.mean(grad_chi_list, axis=0)
        grad_miu = jnp.mean(grad_miu_list, axis=0)

        avg_energy_diff = jnp.mean(energy_diffs)
        avg_true_error = jnp.mean(true_errors)

        # AdamW step (optax convention: grads are for a loss we minimize)
        grads = (grad_chi, grad_miu)
        updates, opt_state = optimizer.update(grads, opt_state, (chi, miu))
        chi_new, miu_new = optax.apply_updates((chi, miu), updates)

        return key, chi_new, miu_new, opt_state, avg_energy_diff, avg_true_error

    # ---------------- Training Loop ----------------
    key = jax.random.PRNGKey(key_seed)

    t0 = time.time()
    energy_history = []
    true_error_history = []
    chi, miu = chi_initial, miu_initial

    for epoch in tqdm(range(n_epochs), desc="Training", unit="epoch"):
        key, chi, miu, opt_state, avg_E, avg_true_error = batched_update(key, chi, miu, opt_state, clamped)
        energy_history.append(avg_E)
        true_error_history.append(avg_true_error)
        if print_energy:
            tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Avg Energy Diff = {avg_E:.6f}")

    t1 = time.time()
    if print_energy:
        print(f"Total training time: {t1 - t0:.3f} s")

    return chi, miu, energy_history, true_error_history


def _plot_grayscale_blocks(values, orientation='horizontal', cmap='gray', figsize=(8, 1), vmin=None, vmax=None, show_colorbar=False):
    """Plot a 1D list/array `values` as adjacent grayscale blocks.

    orientation: 'horizontal' (1 x n) or 'vertical' (n x 1)
    """
    arr = np.asarray(values, dtype=float)
    if vmin is None:
        vmin = np.nanmin(arr)
    if vmax is None:
        vmax = np.nanmax(arr)
    if orientation == 'horizontal':
        im = arr.reshape(1, -1)
        fig, ax = plt.subplots(figsize=figsize)
        mappable = ax.imshow(im, cmap=cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_yticks([])
        ax.set_xticks([])
    else:
        im = arr.reshape(-1, 1)
        fig, ax = plt.subplots(figsize=(figsize[1], figsize[0]) if isinstance(figsize, tuple) else (1, len(arr) / 4))
        mappable = ax.imshow(im, cmap=cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    if show_colorbar:
        plt.colorbar(mappable=mappable, ax=ax, orientation='horizontal' if orientation == 'horizontal' else 'vertical')
    plt.tight_layout()
    plt.show()


def CHL_training_hidden(target_memories, n_species, n_epochs, clamped=None,
                         chi_prev=None, miu_prev=None,
                         gamma_learning=0.5, n_sample=50, p_boundary=0.3,
                         width=0.05, dt_dynamics=1e-1, max_steps=30000, t_end=300.0,
                         verbose=True, seed=None):
    """Train chi/miu via contrastive Hebbian learning.

    If `chi_prev`/`miu_prev` are given (e.g. from a previous call), training
    continues from those values instead of the Hebbian initial guess.
    """
    if seed is None:
        seed = np.random.randint(0, 1000)
    if verbose:
        print("seed:", seed)

    rng = np.random.default_rng(seed)

    # average target memories as hebbian initial guess
    chi_initial = -target_memories.T @ target_memories / target_memories.shape[0]
    np.fill_diagonal(chi_initial, 0.0)
    if chi_prev is not None:
        chi_initial = chi_prev

    if verbose:
        plt.figure(figsize=(8, 5))
        plt.imshow(chi_initial, cmap='coolwarm', interpolation='nearest', vmin=-np.max(np.abs(chi_initial)), vmax=np.max(np.abs(chi_initial)))
        plt.colorbar()
        plt.title('Initial Interaction Matrix')
        plt.show()

    miu_initial = rng.normal(loc=0.0, size=(n_species)) * 0.0
    if miu_prev is not None:
        miu_initial = miu_prev

    chi_learned, miu_learned, energy_diff_hist, true_error_hist = daydreaming_contrastive_hebbian_learning_lagrange_JAX_fast_sample_adamw(
        target_memories=target_memories,
        chi_initial=chi_initial,
        miu_initial=miu_initial,
        V=1.0,
        dt_dynamics=dt_dynamics,
        n_steps_dynamics=max_steps,
        gamma_learning=gamma_learning,
        n_epochs=n_epochs,
        n_sample=n_sample,
        sample_random_each_epoch=False,
        p_boundary=p_boundary,
        width=width,
        t_end=t_end,
        clamped=clamped,
        print_energy=False,
        key_seed=seed,
    )

    if verbose:
        plt.figure(figsize=(8, 5))
        plt.imshow(chi_learned, cmap='coolwarm', interpolation='nearest', vmin=-np.max(np.abs(chi_learned)), vmax=np.max(np.abs(chi_learned)))
        plt.colorbar()
        plt.title('Learned Interaction Matrix')
        plt.show()
        chi_change = chi_learned - chi_initial
        plt.figure(figsize=(8, 5))
        plt.imshow(chi_change, cmap='coolwarm', interpolation='nearest', vmin=-np.max(np.abs(chi_change)), vmax=np.max(np.abs(chi_change)))
        plt.colorbar()
        plt.title('Change in Interaction Matrix')
        plt.show()

        _plot_grayscale_blocks(miu_learned, orientation='horizontal', figsize=(chi_learned.shape[0], 1.0), show_colorbar=True)

        plt.figure(figsize=(8, 5))
        plt.plot(energy_diff_hist)
        plt.xlabel('Epoch')
        plt.ylabel('Average Energy Difference (F+ - F-)')
        plt.title('Contrastive Hebbian Learning Progress')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(true_error_hist)
        plt.xlabel('Epoch')
        plt.ylabel('Average True Error')
        plt.title('Contrastive Hebbian Learning Progress')
        plt.grid(True)
        plt.show()

        print("chi_learned:", chi_learned)
        print("miu_learned:", miu_learned)

    return chi_learned, miu_learned, energy_diff_hist, true_error_hist


# ---------------- Retrieval / recall evaluation ----------------

@partial(jax.jit, static_argnums=4)
def retrieval_jax_core(key, memory, chi_learned, mu_learned, n_flip):
    key, subkey, key_flip_val = jax.random.split(key, 3)
    memory_plus = memory
    memory_test = memory

    flip_indices = jax.random.choice(subkey, memory_test.shape[0], (n_flip,), replace=False)
    random_values = jax.random.randint(
        key_flip_val,
        shape=(n_flip,),
        minval=0,
        maxval=2
    )
    memory_test = memory_test.at[flip_indices].set(random_values)

    beta = 0.5
    memory_test = jnp.exp(memory_test / beta)
    memory_test = memory_test / jnp.sum(memory_test)

    memory_minus_diffrax = forward_sim_x_ssolvent_clamp(
        phi0=memory_test,
        chi=chi_learned,
        mu=mu_learned,
        clamp=0,  # fully free for negative phase
        t_end=500.0,
        dt=0.1,
        samples=10,
        max_steps=30000,
    )
    memory_minus = x_to_phi(memory_minus_diffrax.ys[-1])

    memory_minus = jnp.log(memory_minus * jnp.exp(1) * jnp.sum(memory_plus))
    memory_minus = jnp.where(memory_minus > 0.0, 1, 0)

    equal = jnp.array_equal(memory_minus, memory_plus)
    return equal, key


def retrieval_batch_for_nflip(key, memories, chi_learned, mu_learned, n_flip, n_runs):
    num_memories = memories.shape[0]

    keys = jax.random.split(key, n_runs * num_memories).reshape((n_runs, num_memories, 2))
    memories_exp = jnp.broadcast_to(memories, (n_runs, num_memories, memories.shape[1]))

    @jax.vmap
    @jax.vmap
    def run_one(key, memory):
        eq, _ = retrieval_jax_core(key, memory, chi_learned, mu_learned, n_flip)
        return eq

    results = run_one(keys, memories_exp)  # shape (n_runs, num_memories)
    retrieval_rate = jnp.mean(results)
    return retrieval_rate


def batch_retrieval_jax_batched(key, memories, chi_learned, mu_learned, alphas=None, n_runs=50, if_plot=False):
    if alphas is None:
        alphas = np.linspace(0, 1, 11)

    memory_len = memories.shape[1]
    n_flips = np.floor(alphas * memory_len).astype(int)  # numpy int array, concrete
    print(n_flips)

    retrieval_rates = []

    for alpha, n_flip in zip(alphas, n_flips):
        key, subkey = jax.random.split(key)
        rate = retrieval_batch_for_nflip(subkey, memories, chi_learned, mu_learned, n_flip, n_runs)
        retrieval_rates.append(rate)

    retrieval_rates = jnp.array(retrieval_rates)

    if if_plot:
        plt.plot(alphas, retrieval_rates)
        plt.xlabel('Flip ratio (alpha)')
        plt.ylabel('Retrieval Rate')
        plt.title('Retrieval Rate vs Flip ratio')
        plt.ylim(0, 1)
        plt.show()

    return retrieval_rates

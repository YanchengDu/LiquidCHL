"""Contrastive Hebbian learning (daydreaming) training for Model A — classifier tasks
(AND, XOR, Circular, ...).

Unlike the target-phase-retrieval and IO variants, classifier tasks have no fixed
set of target memories: training memories are generated on the fly by a
task-specific `make_random_memories_fn(key, sample_size, n_species) -> (key, mems)`
that encodes the input/output boundary for that task (e.g. AND's quadrant rule,
XOR's parity rule, Circular's radius rule). Pass that function in to
`daydreaming_contrastive_hebbian_learning_lagrange_JAX_fast_sample_adamw` and/or
`CHL_training_hidden`.

`CHL_training_hidden` starts from a random interaction matrix (like the IO
variant) and supports continuing training from a previous result via
`chi_prev`/`miu_prev`.
"""

import time

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import optax
from tqdm.auto import tqdm

from Dynamics.Model_A import forward_sim_x_ssolvent_clamp, x_to_phi


def daydreaming_contrastive_hebbian_learning_lagrange_JAX_fast_sample_adamw(
    target_memories, chi_initial, miu_initial,
    make_random_memories_fn=None,
    V=1.0, gamma_dynamics=1.0, dt_dynamics=1e-2, n_steps_dynamics=30000,
    gamma_learning=10.0, n_epochs=200, clamped=None,
    n_sample=None, sample_random_each_epoch=False,
    t_end=300.0,
    weight_decay=1e-5,
    print_energy=True, key_seed=42
):
    """Train chi/miu via contrastive Hebbian learning (daydreaming) with AdamW.

    If `target_memories` is None or `sample_random_each_epoch` is True, training
    memories are generated each epoch via `make_random_memories_fn(key, sample_size,
    n_species) -> (key, mems)` (a task-specific callable encoding the classifier's
    input/output boundary). Otherwise, memories are sampled (with replacement if
    needed) from the fixed `target_memories`.
    """
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
        # Positive phase
        phi_ini = jax.random.normal(key, shape=phi.shape) * 0.1 * 1 / N + 1 / N
        phi_ini = phi_ini.at[:clamped].set(phi[:clamped])
        phi_ini = jnp.clip(phi_ini, 1e-8, 1.0)
        sum_fixed = jnp.sum(phi_ini[:clamped])
        sum_rem = jnp.sum(phi_ini[clamped:])
        phi_ini = phi_ini.at[clamped:].set(phi_ini[clamped:] / sum_rem * (1 - sum_fixed))
        phi_plus_diffrax = forward_sim_x_ssolvent_clamp(
            phi0=phi_ini,
            chi=chi,
            mu=miu,
            clamp=clamped,
            t_end=t_end,
            dt=dt_dynamics,
            samples=10,
            max_steps=n_steps_dynamics,
        )
        phi_plus = x_to_phi(phi_plus_diffrax.ys[-1])

        # Negative phase
        phi_minus_diffrax = forward_sim_x_ssolvent_clamp(
            phi0=phi_plus,
            chi=chi,
            mu=miu,
            clamp=2,
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

        # Gradients
        true_error = compute_true_error_swap(phi_ini, phi_minus, N)
        grad_scaling = 1.0
        grad_chi = (jnp.outer(phi_plus, phi_plus) - jnp.outer(phi_minus, phi_minus)) * grad_scaling
        grad_miu = (phi_plus - phi_minus) * grad_scaling

        # Zero out fixed elements
        grad_chi = grad_chi.at[jnp.diag_indices(grad_chi.shape[0])].set(0.0)
        grad_miu = grad_miu.at[0].set(0.0).at[1].set(0.0).at[-1].set(0.0)
        # zero out last row and column (solvent species), and the two input species
        grad_chi = grad_chi.at[-1, :].set(0.0)
        grad_chi = grad_chi.at[:, -1].set(0.0)
        grad_chi = grad_chi.at[0, 1].set(0.0)
        grad_chi = grad_chi.at[1, 0].set(0.0)

        return grad_chi, grad_miu, energy_diff, true_error

    # ---------------- AdamW optimizer ----------------
    params = (chi_initial, miu_initial)
    optimizer = optax.adamw(learning_rate=gamma_learning, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    # ---------------- Batched Update ----------------
    def batched_update(key, chi, miu, opt_state, clamped):
        sample_size = n_sample if n_sample is not None else (target_memories.shape[0] if target_memories is not None else 1)
        N = chi.shape[0]
        if sample_random_each_epoch or (target_memories is None):
            key, sampled_memories = make_random_memories_fn(key, sample_size, N)
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


def CHL_training_hidden(target_memories, n_species, n_epochs, make_random_memories_fn,
                         clamped=None, chi_prev=None, miu_prev=None,
                         gamma_learning=0.5, n_sample=50, sample_random_each_epoch=False,
                         dt_dynamics=1e-1, max_steps=30000, t_end=300.0,
                         verbose=True, seed=None):
    """Train chi/miu via contrastive Hebbian learning for a classifier task.

    `chi_initial` is a random interaction matrix (diagonal, solvent row/column,
    and the two input species' cross-term fixed at zero). If `chi_prev`/`miu_prev`
    are given (e.g. from a previous call), training continues from those values
    instead of this random initial guess.

    `make_random_memories_fn(key, sample_size, n_species) -> (key, mems)` supplies
    the task-specific training-memory distribution (e.g. AND/XOR/Circular boundary
    rules), used when `target_memories` is None or `sample_random_each_epoch=True`.
    """
    if seed is None:
        seed = np.random.randint(0, 1000)
    if verbose:
        print("seed:", seed)

    rng = np.random.default_rng(seed)

    chi_initial = rng.uniform(low=-15.0, high=15.0, size=(n_species, n_species))
    chi_initial[:4, :4] = rng.normal(loc=0.0, size=(4, 4)) * 0.1
    chi_initial = (chi_initial + chi_initial.T) / 2
    np.fill_diagonal(chi_initial, 0.0)
    chi_initial[-1, :] = 0.0
    chi_initial[:, -1] = 0.0
    chi_initial[0, 1] = 0.0
    chi_initial[1, 0] = 0.0
    if chi_prev is not None:
        chi_initial = chi_prev

    if verbose:
        plt.figure(figsize=(8, 5))
        plt.imshow(chi_initial, cmap='coolwarm', interpolation='nearest', vmin=-np.max(np.abs(chi_initial)), vmax=np.max(np.abs(chi_initial)))
        plt.colorbar()
        plt.title('Initial Interaction Matrix')
        plt.show()

    miu_initial = rng.normal(loc=0.0, size=(n_species)) * 0.0
    miu_initial[0] = miu_initial[1] = miu_initial[-1] = 0.0
    if miu_prev is not None:
        miu_initial = miu_prev

    chi_learned, miu_learned, energy_diff_hist, true_error_hist = daydreaming_contrastive_hebbian_learning_lagrange_JAX_fast_sample_adamw(
        target_memories=target_memories,
        chi_initial=chi_initial,
        miu_initial=miu_initial,
        make_random_memories_fn=make_random_memories_fn,
        V=1.0,
        dt_dynamics=dt_dynamics,
        n_steps_dynamics=max_steps,
        gamma_learning=gamma_learning,
        n_epochs=n_epochs,
        n_sample=n_sample,
        sample_random_each_epoch=sample_random_each_epoch,
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


def evaluate_and_plot_training(
    chi_learned, miu_learned,
    forward_sim_x_ssolvent_clamp,
    energy_diff_hist=None,
    true_error_hist=None,
    n_species=17,
    clamped=2,
    n_points=1000,
    dt_dynamics=None,
    max_steps=None,
    t_end=300.0,
    key_seed=0,
    use_log=False,
    vmin=0.0,
    vmax=0.25,
    plot=True,
    is_above_boundary=None,
):
    """
    Single sampling run: (1) training summary, (2) test accuracy vs the task's
    input/output boundary, (3) scatter plots (Output1, Output2, Hidden...). Reuses
    the same n_points for evaluation and plotting to avoid running the forward sim
    twice. Set plot=False to return metrics only (for parameter sweeps).

    `is_above_boundary(x, y) -> bool array` is the task-specific ground-truth rule
    (e.g. AND's quadrant rule, XOR's parity rule, Circular's radius rule) used for
    the accuracy report. If None, accuracy reporting is skipped.
    """
    if dt_dynamics is None:
        dt_dynamics = 1e-1
    if max_steps is None:
        max_steps = 30000
    N = n_species
    key = jax.random.PRNGKey(key_seed)
    key_x, key_y, key_rand = jax.random.split(key, 3)

    x = jax.random.uniform(key_x, shape=(n_points,), minval=1e-3, maxval=0.25)
    y = jax.random.uniform(key_y, shape=(n_points,), minval=1e-3, maxval=0.25)
    rand_tail = jax.random.normal(key_rand, shape=(n_points, N)) * 0.1 / N + 1 / N

    def make_input(xi, yi, rand_row):
        phi = rand_row.at[:2].set(jnp.array([xi, yi]))
        phi = jnp.clip(phi, 1e-8, 1.0)
        sum_fixed = jnp.sum(phi[:2])
        sum_rem = jnp.sum(phi[2:])
        phi = phi.at[2:].set(phi[2:] / sum_rem * (1 - sum_fixed))
        return phi

    inputs = jax.vmap(make_input)(x, y, rand_tail)

    simulate_batched = jax.vmap(
        lambda phi0: x_to_phi(forward_sim_x_ssolvent_clamp(
            phi0=phi0, chi=chi_learned, mu=miu_learned, clamp=clamped,
            t_end=t_end, dt=dt_dynamics, samples=10, max_steps=max_steps,
        ).ys[-1])
    )
    outputs = simulate_batched(inputs)
    if hasattr(outputs, "block_until_ready"):
        outputs.block_until_ready()

    outputs_np = np.array(outputs)
    x_np = np.array(x)
    y_np = np.array(y)

    # ----- 1. Training curve summary -----
    if plot and energy_diff_hist is not None and true_error_hist is not None:
        energy_hist = np.array(energy_diff_hist)
        error_hist = np.array(true_error_hist)
        print("=== Training summary ===")
        print(f"  Energy diff (F+ - F-):  initial = {energy_hist[0]:.6f},  final = {energy_hist[-1]:.6f}")
        print(f"  True error (swap loss):  initial = {error_hist[0]:.6f},  final = {error_hist[-1]:.6f}")
        if len(energy_hist) > 1:
            print(f"  Energy trend: {'↓' if energy_hist[-1] < energy_hist[0] else '↑'}  (min = {energy_hist.min():.6f})")
        if len(error_hist) > 1:
            print(f"  Error trend:  {'↓' if error_hist[-1] < error_hist[0] else '↑'}  (min = {error_hist.min():.6f})")
        print()

    # ----- 2. Test accuracy (same samples as plots) -----
    eval_metrics = {"n_test": n_points}
    if is_above_boundary is not None:
        true_above = is_above_boundary(x_np, y_np)
        pred_above = (outputs_np[:, 2] > outputs_np[:, 3]).astype(int)
        accuracy = np.mean(pred_above == true_above)
        n_above, n_below = true_above.sum(), len(true_above) - true_above.sum()
        correct_above = np.sum((pred_above == 1) & (true_above == 1))
        correct_below = np.sum((pred_above == 0) & (true_above == 0))

        if plot:
            print("=== Test accuracy (above vs below boundary) ===")
            print(f"  Overall accuracy: {accuracy*100:.2f}%  ({int(np.round(accuracy*n_points))}/{n_points} correct)")
            if n_above > 0:
                print(f"  Above boundary:   {correct_above}/{n_above} correct  ({100*correct_above/n_above:.1f}%)")
            if n_below > 0:
                print(f"  Below boundary:   {correct_below}/{n_below} correct  ({100*correct_below/n_below:.1f}%)")
            print()

        eval_metrics.update({
            "accuracy": float(accuracy),
            "correct_above": int(correct_above),
            "correct_below": int(correct_below),
            "n_above": int(n_above),
            "n_below": int(n_below),
        })

    # ----- 3. Scatter plots (same outputs_np) -----
    record_list = [outputs_np[:, i] for i in range(2, n_species - 1)]
    if plot:
        titlelist = ["Output1", "Output2"] + [f"Hidden{i}" for i in range(1, n_species - 4)]
        n_plots = len(record_list)
        n_rows = max(1, int((n_species - 3) / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows), constrained_layout=True)
        axes = np.atleast_1d(axes).flatten()
        for i, ax in enumerate(axes):
            if i >= n_plots:
                ax.set_visible(False)
                continue
            data = record_list[i]
            norm = colors.LogNorm(vmin=max(1e-6, data.min()), vmax=max(1e-6, data.max())) if use_log else colors.Normalize(vmin=vmin, vmax=vmax)
            im = ax.scatter(x_np, y_np, c=data, cmap='coolwarm', s=20, linewidths=0)
            fig.colorbar(im, ax=ax).set_label(titlelist[i])
            ax.set_xlabel("phi₀")
            ax.set_ylabel("phi₁")
            ax.set_title(titlelist[i])
            ax.grid(False)
        plt.show()

    results = {"x": x_np, "y": y_np, "output1": record_list[0], "output2": record_list[1], "hidden1": record_list[2], "hidden2": record_list[3]}
    return eval_metrics, results, outputs_np

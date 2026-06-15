"""Contrastive Hebbian learning for spatial (Model B) Cahn-Hilliard systems.

Training loop that updates the interaction matrix chi using a contrastive
Hebbian rule: run the PDE dynamics to convergence from a random initial
condition, then apply a Hebbian-like update based on the pairwise spatial
correlations of the final field.

Public API
----------
train_contrastive_hebbian_newscheme(training_memories, ...)
    -> chi_update, J_history, phi_final_list
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import tqdm

from Dynamics.Spatial_Cahn_Hilliard import (
    init_phi_multi_jax,
    compute_energy_spectral_jax,
    run_cahn_hilliard_multi_FH_rstab_jax_until_converged,
    plot_energy,
)


def train_contrastive_hebbian_newscheme(
    training_memories,
    chi_ref=None,
    chi_init=None,
    phi_initial=None,
    Epochs=50,
    nc=17,
    nx=128,
    ny=128,
    kappa_val=5.0,
    dt=1e-3,
    noise_amp=0.003,
    save_interval=200,
    r_stab=4.0,
    chi_clip=25.0,
    lr=0.1,
    plot=True,
    seed=42,
    sigma_spatial=5.0,
    max_steps=200000,
    max_time=200.0,
    dt_min=1e-6,
    dt_max=0.10,
    energy_increase_tol=1e-6,
    mean_tol=1e-5,
    raw_mass_tol=1e-5,
    raw_min_tol=-1e-6,
    field_tol=1e-5,
    energy_slope_tol=1e-2,
    window=20,
    min_steps_check=1000,
    grow_dt=1.05,
    track_energy=True,
    plot_gap=10,
):
    """Contrastive-Hebbian training loop for the spatial Cahn-Hilliard system.

    Runs PDE dynamics to convergence from a random initial composition, then
    updates chi via a Hebbian-like rule using the pairwise spatial correlations
    of the final field and the reference matrix chi_ref.

    Parameters
    ----------
    training_memories : array, shape (K, nc)
        Target memory vectors (partitioned, i.e. sum to 1).
    chi_ref : list of (nc, nc) arrays
        Reference Hebbian matrices, one per memory. Used to compute the
        Hebbian update direction.
    chi_init : (nc, nc) array or None
        Initial interaction matrix. Defaults to mean of chi_ref.
    phi_initial : ignored (reserved for future use)
    Epochs : int
    nc, nx, ny : int
        Number of components and spatial grid size.
    kappa_val : float
        Gradient penalty coefficient (uniform across components).
    dt : float
        Initial time step.
    noise_amp : float
    save_interval : int
        How often (steps) to save frames when plot=True.
    r_stab : float
        r-stabilization parameter.
    chi_clip : float
        Clip chi entries to [-chi_clip, chi_clip] after each update.
    lr : float
        Learning rate for chi update.
    plot : bool
        Whether to generate plots every plot_gap epochs.
    seed : int or None
    sigma_spatial : float
        Smoothing length for initial condition.
    max_steps, max_time, dt_min, dt_max : convergence/time limits.
    energy_increase_tol, mean_tol, raw_mass_tol, raw_min_tol,
    field_tol, energy_slope_tol, window, min_steps_check, grow_dt :
        Convergence criteria (passed through to the dynamics runner).
    track_energy : bool
    plot_gap : int
        Plot every this many epochs.

    Returns
    -------
    chi_update : (nc, nc) ndarray
    J_history  : list of float  (mean ΔF per epoch)
    phi_final_list : list of (nc, nx, ny) ndarrays
    """
    if chi_init is None:
        chi_init = np.mean(chi_ref, axis=0)
    chi_update = np.array(chi_init, dtype=np.float64, copy=True)
    kappa = np.full(nc, kappa_val, dtype=np.float64)
    dx = dy = 1.0

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    K2 = jnp.asarray(kx[:, None] ** 2 + ky[None, :] ** 2, dtype=jnp.float64)

    J_history = []
    phi_final_list = []

    for epoch in tqdm.tqdm(range(Epochs), desc="Training"):
        epoch_seed = np.random.randint(0, 1000) if seed is None else seed + epoch
        use_training_mode = not plot
        delta_record = []
        J_record = []

        phi_initial_list = [
            init_phi_multi_jax(
                nc=nc, nx=nx, ny=ny,
                mean_phi=np.mean(np.array(training_memories), axis=0),
                noise_amp=noise_amp,
                seed=epoch_seed,
                sigma_spatial=sigma_spatial,
            )
        ]

        for pos, phi_ini in enumerate(phi_initial_list):
            mean_phi = np.mean(phi_ini, axis=(1, 2))
            print("pos:", pos)
            sim_out = run_cahn_hilliard_multi_FH_rstab_jax_until_converged(
                nc=nc, nx=nx, ny=ny, dx=dx, dy=dy,
                M=jnp.ones(nc, dtype=jnp.float64),
                kappa=jnp.asarray(kappa, dtype=jnp.float64),
                chi=jnp.asarray(chi_update, dtype=jnp.float64),
                nu=None, dt=dt, mean_phi=mean_phi,
                initial_phi=phi_ini,
                noise_amp=noise_amp / nc,
                save_interval=save_interval,
                seed=epoch_seed, verbose=False,
                track_energy=track_energy,
                training=use_training_mode,
                r_stab=r_stab, max_steps=max_steps, max_time=max_time,
                dt_min=dt_min, dt_max=dt_max,
                energy_increase_tol=energy_increase_tol,
                mean_tol=mean_tol, raw_mass_tol=raw_mass_tol,
                raw_min_tol=raw_min_tol, field_tol=field_tol,
                energy_slope_tol=energy_slope_tol,
                window=window, min_steps_check=min_steps_check,
                grow_dt=grow_dt, sigma_spatial=sigma_spatial,
            )

            if use_training_mode:
                phi_final = sim_out
                frames = energies_out = time_saved_out = None
            else:
                frames, phi_final, _, energies_out, _, time_saved_out = sim_out

            F_ini = compute_energy_spectral_jax(
                jnp.asarray(phi_ini, dtype=jnp.float64),
                jnp.asarray(chi_update, dtype=jnp.float64),
                jnp.zeros(nc, dtype=jnp.float64),
                K2, dx, dy, None,
            )
            F_final = compute_energy_spectral_jax(
                jnp.asarray(phi_final, dtype=jnp.float64),
                jnp.asarray(chi_update, dtype=jnp.float64),
                jnp.zeros(nc, dtype=jnp.float64),
                K2, dx, dy, None,
            )

            J = float(F_ini - F_final)
            J_record.append(J)
            phi_final_np = np.array(phi_final)
            phi_final_list.append(phi_final_np)

            if plot and (epoch % plot_gap == 0 or epoch == Epochs - 1):
                print(f"Epoch {epoch + 1}/{Epochs}, ΔF = {J:.6f}")

                plt.figure(figsize=(8, 5))
                vmax = np.max(np.abs(chi_update)) + 1e-12
                plt.imshow(chi_update, cmap="coolwarm", interpolation="nearest",
                           vmin=-vmax, vmax=vmax)
                plt.colorbar()
                plt.title(f"Interaction Matrix after epoch {epoch + 1}")
                plt.tight_layout()
                plt.show()

                final_frame = frames[-1] if (frames is not None and len(frames) > 0) else phi_final_np
                n_show = min(nc, final_frame.shape[0])
                ncols = 4
                nrows = int(np.ceil(n_show / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
                axes = np.atleast_1d(axes).ravel()
                for idx in range(n_show):
                    im = axes[idx].imshow(final_frame[idx], origin="lower")
                    axes[idx].set_title(f"comp {idx} final")
                    axes[idx].axis("off")
                    plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
                for ax in axes[n_show:]:
                    ax.axis("off")
                plt.tight_layout()
                plt.show()

                if track_energy and energies_out is not None and time_saved_out is not None:
                    plot_energy(time_saved_out, energies_out, title=f"Energy decay (r={r_stab})")

            # Hebbian update: chi_ref + spatial correlations of final field
            corr = np.mean(
                np.einsum("ixy,jxy->ijxy", phi_final_np, phi_final_np),
                axis=(2, 3),
            )
            delta = np.array(chi_ref[pos], dtype=np.float64) + corr
            np.fill_diagonal(delta, 0.0)
            delta_record.append(delta)

        if J_history and np.mean(J_record) > J_history[-1]:
            print(f"Warning: J increased in epoch {epoch + 1} "
                  f"(ΔF = {np.mean(J_record):.6f} vs previous {J_history[-1]:.6f})")

        J_history.append(np.mean(J_record))
        delta_mean = np.mean(delta_record, axis=0)
        chi_update = chi_update + lr * delta_mean / (np.max(np.abs(delta_mean)) + 1e-12)
        np.fill_diagonal(chi_update, 0.0)
        chi_update = np.clip(chi_update, -chi_clip, chi_clip)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(J_history)
        plt.title("J_history")
        plt.xlabel("Epoch")
        plt.ylabel("ΔF")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return chi_update, J_history, phi_final_list

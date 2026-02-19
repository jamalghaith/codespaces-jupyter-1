#!/usr/bin/env python3
"""KS / KPZ-style surface evolution simulator (1D & 2D) with optional
GPU (CuPy) acceleration and time-dependent coefficient modulation from
incidence angle and sample rotation. Extracted from notebook version.

Usage (basic 2D CPU run):
    python ks_sim.py --dim 2 --steps 2000 --Nx 256 --Ny 256

Attempt GPU (requires cupy):
    python ks_sim.py --dim 2 --backend cupy --steps 2000

Save MP4 animation (requires ffmpeg installed):
    python ks_sim.py --dim 2 --save-anim --anim-file surface.mp4 --steps 4000

Parameter modulation example (angle 45°, fast rotation):
    python ks_sim.py --dim 2 --theta 45 --rotation-rate 5.0 --use-rotation \
        --lam-par 4.0 --lam-perp 2.0 --nu-par 1.2 --nu-perp 0.8

Notes:
- If cupy not installed or no GPU, backend falls back to numpy automatically unless
  backend is explicitly forced to 'cupy'.
- The code keeps arrays in GPU memory (if using CuPy) and only transfers to host
  when saving animation frames or printing final summaries.
- FFT sizes: powers of two are fastest.
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Optional

try:
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------------------------------------------------
# Data container
# ------------------------------------------------------------
@dataclass
class Params:
    # Base coefficients
    v0: float = 0.05
    gamma: float = 0.0
    nu: float = 1.0
    B: float = 1.0
    lam: float = 3.0
    D: float = 0.2
    # Domain
    Lx: float = 256.0
    Ly: float = 256.0
    Nx: int = 256
    Ny: int = 256
    # Time integration
    dt: float = 0.01
    steps: int = 4000
    output_interval: int = 50
    # Numerical options
    dealias: bool = True
    seed: int = 0
    dim: int = 2  # 1 or 2
    # Angle / rotation controls
    theta_deg: float = 0.0
    phi0_deg: float = 0.0
    rotation_rate: float = 0.0  # radians per unit time
    use_rotation: bool = False
    # Anisotropy collapsed each step (optional)
    lam_parallel: Optional[float] = None
    lam_perp: Optional[float] = None
    nu_parallel: Optional[float] = None
    nu_perp: Optional[float] = None
    B_parallel: Optional[float] = None
    B_perp: Optional[float] = None
    v0_angle_mode: str = "cos"  # cos|invcos|none
    # Animation / output
    save_anim: bool = False
    anim_file: str = "surface.mp4"
    anim_fps: int = 20
    # Backend
    backend: str = "auto"  # auto|numpy|cupy
    # Runtime toggles
    verbose: bool = True

# ------------------------------------------------------------
# Backend abstraction
# ------------------------------------------------------------
class XP:
    """Namespace holding numpy or cupy chosen backend."""
    xp = np  # default

    @staticmethod
    def use(backend: str):
        if backend == "numpy":
            XP.xp = np
        elif backend == "cupy":
            if not _CUPY_AVAILABLE:
                raise RuntimeError("CuPy backend requested but CuPy not available")
            XP.xp = cp  # type: ignore
        elif backend == "auto":
            if _CUPY_AVAILABLE:
                XP.xp = cp  # type: ignore
            else:
                XP.xp = np
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @staticmethod
    def is_gpu():
        return _CUPY_AVAILABLE and XP.xp is not np

# Convenience alias after initialization

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def spectral_grid_1d(Lx, Nx, xp):
    dx = Lx / Nx
    k = 2.0 * math.pi * xp.fft.fftfreq(Nx, d=dx)
    return k, dx

def spectral_grid_2d(Lx, Ly, Nx, Ny, xp):
    dx = Lx / Nx
    dy = Ly / Ny
    kx = 2.0 * math.pi * xp.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * math.pi * xp.fft.fftfreq(Ny, d=dy)
    kx_grid, ky_grid = xp.meshgrid(kx, ky, indexing='ij')
    k2 = kx_grid**2 + ky_grid**2
    return kx_grid, ky_grid, k2, dx, dy

def dealias_mask_1d(k, xp):
    kmax = xp.max(xp.abs(k))
    cutoff = (2.0/3.0) * kmax
    return (xp.abs(k) <= cutoff).astype(float)

def dealias_mask_2d(kx, ky, xp):
    kxmax = xp.max(xp.abs(kx))
    kymax = xp.max(xp.abs(ky))
    kx_cut = (2.0/3.0) * kxmax
    ky_cut = (2.0/3.0) * kymax
    return ((xp.abs(kx) <= kx_cut) & (xp.abs(ky) <= ky_cut)).astype(float)

# ------------------------------------------------------------
# Coefficient modulation
# ------------------------------------------------------------

def eff_coef(par, perp, default, phi, xp):
    if par is None or perp is None or abs(par - perp) < 1e-14:
        return default
    c = xp.cos(phi)
    return perp + (par - perp) * (c * c)

# ------------------------------------------------------------
# 1D simulation core (no time-dependent modulation needed typically)
# ------------------------------------------------------------

def run_1d(p: Params):
    xp = XP.xp
    rng = np.random.default_rng(p.seed)  # CPU RNG (noise copied to GPU if needed)
    k, dx = spectral_grid_1d(p.Lx, p.Nx, xp)
    k2 = k**2
    k4 = k2**2
    Lk = -p.gamma - p.nu * k2 - p.B * k4
    mask = dealias_mask_1d(k, xp) if p.dealias else 1.0
    h = 0.01 * rng.standard_normal(p.Nx)
    if XP.is_gpu():
        h = cp.asarray(h)  # type: ignore
    denom = (1.0 - p.dt * Lk)
    times = []
    roughness = []
    n_frames = p.steps // p.output_interval
    for frame in range(n_frames):
        for _ in range(p.output_interval):
            h_hat = xp.fft.fft(h)
            dhdx = xp.fft.ifft(1j * k * h_hat).real
            nonlinear = 0.5 * p.lam * (dhdx**2)
            if p.D > 0:
                eta_host = math.sqrt(2 * p.D / (dx * p.dt)) * rng.standard_normal(p.Nx)
                eta = xp.asarray(eta_host) if XP.is_gpu() else eta_host
            else:
                eta = 0.0
            rhs_real = nonlinear - p.v0 + eta
            rhs_hat = xp.fft.fft(rhs_real)
            if p.dealias:
                rhs_hat *= mask
            h_hat_new = (h_hat + p.dt * rhs_hat) / denom
            if p.dealias:
                h_hat_new *= mask
            h = xp.fft.ifft(h_hat_new).real
        t = (frame+1) * p.output_interval * p.dt
        W = xp.sqrt(xp.mean((h - xp.mean(h))**2))
        times.append(t)
        roughness.append(float(W.get() if XP.is_gpu() else W))
        if p.verbose and (frame % max(1, n_frames//10) == 0):
            print(f"[1D] Frame {frame+1}/{n_frames} t={t:.2f} W={roughness[-1]:.4f}")
    # Return final surface and roughness curve
    h_host = h.get() if XP.is_gpu() else h
    return h_host, np.array(times), np.array(roughness)

# ------------------------------------------------------------
# 2D simulation with time-dependent coefficients
# ------------------------------------------------------------

def run_2d(p: Params, collect_frames: bool = False):
    xp = XP.xp
    rng = np.random.default_rng(p.seed)
    kx, ky, k2, dx, dy = spectral_grid_2d(p.Lx, p.Ly, p.Nx, p.Ny, xp)
    k4 = k2**2
    mask = dealias_mask_2d(kx, ky, xp) if p.dealias else 1.0
    h0 = 0.01 * rng.standard_normal((p.Nx, p.Ny))
    h = xp.asarray(h0) if XP.is_gpu() else h0
    base_v0 = p.v0; base_nu = p.nu; base_B = p.B; base_lam = p.lam
    theta = math.radians(p.theta_deg)
    cos_theta = math.cos(theta)

    frames = []  # optional list of surfaces (host)
    times = []
    roughness = []
    step_count = 0
    n_frames = p.steps // p.output_interval

    for frame in range(n_frames):
        for _ in range(p.output_interval):
            t_curr = step_count * p.dt
            phi = math.radians(p.phi0_deg)
            if p.use_rotation:
                phi += p.rotation_rate * t_curr
            phi_xp = xp.array(phi)
            lam_eff = eff_coef(p.lam_parallel, p.lam_perp, base_lam, phi_xp, xp)
            nu_eff = eff_coef(p.nu_parallel, p.nu_perp, base_nu, phi_xp, xp)
            B_eff = eff_coef(p.B_parallel, p.B_perp, base_B, phi_xp, xp)
            if p.v0_angle_mode == "cos":
                v0_eff = base_v0 * cos_theta
            elif p.v0_angle_mode == "invcos" and cos_theta > 1e-12:
                v0_eff = base_v0 / cos_theta
            else:
                v0_eff = base_v0
            h_hat = xp.fft.fftn(h)
            Lk = -p.gamma - nu_eff * k2 - B_eff * k4
            denom = (1.0 - p.dt * Lk)
            dhdx = xp.fft.ifftn(1j * kx * h_hat).real
            dhdy = xp.fft.ifftn(1j * ky * h_hat).real
            nonlinear = 0.5 * lam_eff * (dhdx**2 + dhdy**2)
            if p.D > 0:
                eta_host = math.sqrt(2 * p.D / (dx * dy * p.dt)) * rng.standard_normal((p.Nx, p.Ny))
                eta = xp.asarray(eta_host) if XP.is_gpu() else eta_host
            else:
                eta = 0.0
            rhs = nonlinear - v0_eff + eta
            rhs_hat = xp.fft.fftn(rhs)
            if p.dealias:
                rhs_hat *= mask
            h_hat_new = (h_hat + p.dt * rhs_hat) / denom
            if p.dealias:
                h_hat_new *= mask
            h = xp.fft.ifftn(h_hat_new).real
            step_count += 1
        t = (frame+1) * p.output_interval * p.dt
        W = xp.sqrt(xp.mean((h - xp.mean(h))**2))
        times.append(t)
        roughness.append(float(W.get() if XP.is_gpu() else W))
        if collect_frames:
            frames.append(h.get() if XP.is_gpu() else h.copy())
        if p.verbose and (frame % max(1, n_frames//10) == 0):
            print(f"[2D] Frame {frame+1}/{n_frames} t={t:.2f} W={roughness[-1]:.4f}")
    h_host = h.get() if XP.is_gpu() else h
    return h_host, np.array(times), np.array(roughness), frames

# ------------------------------------------------------------
# Animation helpers (done on CPU memory)
# ------------------------------------------------------------

def save_animation_2d(frames, p: Params):  # pragma: no cover - I/O heavy
    if not frames:
        print("No frames collected for animation.")
        return
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(frames[0].T, origin='lower', cmap='viridis',
                   extent=[0, p.Lx, 0, p.Ly], aspect='auto')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('2D Surface Evolution')
    plt.colorbar(im, ax=ax)
    def update(i):
        im.set_data(frames[i].T)
        im.set_clim(vmin=frames[i].min(), vmax=frames[i].max())
        ax.set_title(f'2D Surface t={ (i+1)*p.output_interval*p.dt:.2f }')
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=p.anim_fps, bitrate=1800)
        ani.save(p.anim_file, writer=writer)
        print(f"Saved animation -> {p.anim_file}")
    except Exception as e:
        print(f"Failed to save animation (install ffmpeg?): {e}")
    finally:
        plt.close(fig)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> Params:
    ap = argparse.ArgumentParser(description="KS/KPZ surface simulator with optional GPU")
    # Basic physical / numerical params
    ap.add_argument('--v0', type=float, default=0.05)
    ap.add_argument('--gamma', type=float, default=0.0)
    ap.add_argument('--nu', type=float, default=1.0)
    ap.add_argument('--B', type=float, default=1.0)
    ap.add_argument('--lam', type=float, default=3.0)
    ap.add_argument('--D', type=float, default=0.2)
    ap.add_argument('--Lx', type=float, default=256.0)
    ap.add_argument('--Ly', type=float, default=256.0)
    ap.add_argument('--Nx', type=int, default=256)
    ap.add_argument('--Ny', type=int, default=256)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--output-interval', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-dealias', action='store_true', help='Disable 2/3 dealiasing')
    ap.add_argument('--dim', type=int, choices=[1,2], default=2)
    # Angle / rotation
    ap.add_argument('--theta', type=float, default=0.0)
    ap.add_argument('--phi0', type=float, default=0.0)
    ap.add_argument('--rotation-rate', type=float, default=0.0)
    ap.add_argument('--use-rotation', action='store_true')
    ap.add_argument('--lam-par', type=float, default=None)
    ap.add_argument('--lam-perp', type=float, default=None)
    ap.add_argument('--nu-par', type=float, default=None)
    ap.add_argument('--nu-perp', type=float, default=None)
    ap.add_argument('--B-par', type=float, default=None)
    ap.add_argument('--B-perp', type=float, default=None)
    ap.add_argument('--v0-angle-mode', type=str, choices=['cos','invcos','none'], default='cos')
    # Output / animation
    ap.add_argument('--save-anim', action='store_true')
    ap.add_argument('--anim-file', type=str, default='surface.mp4')
    ap.add_argument('--anim-fps', type=int, default=20)
    # Backend
    ap.add_argument('--backend', type=str, choices=['auto','numpy','cupy'], default='auto')
    ap.add_argument('--quiet', action='store_true')

    args = ap.parse_args()
    p = Params(
        v0=args.v0, gamma=args.gamma, nu=args.nu, B=args.B, lam=args.lam, D=args.D,
        Lx=args.Lx, Ly=args.Ly, Nx=args.Nx, Ny=args.Ny, dt=args.dt, steps=args.steps,
        output_interval=args.output_interval, seed=args.seed, dealias=not args.no_dealias,
        dim=args.dim, theta_deg=args.theta, phi0_deg=args.phi0, rotation_rate=args.rotation_rate,
        use_rotation=args.use_rotation, lam_parallel=args.lam_par, lam_perp=args.lam_perp,
        nu_parallel=args.nu_par, nu_perp=args.nu_perp, B_parallel=args.B_par, B_perp=args.B_perp,
        v0_angle_mode=args.v0_angle_mode, save_anim=args.save_anim, anim_file=args.anim_file,
        anim_fps=args.anim_fps, backend=args.backend, verbose=not args.quiet
    )
    return p

# ------------------------------------------------------------
# Main run
# ------------------------------------------------------------

def main():  # pragma: no cover - CLI
    p = parse_args()
    XP.use(p.backend)
    if p.verbose:
        print(f"Backend: {'CuPy (GPU)' if XP.is_gpu() else 'NumPy (CPU)'}")
        print(p)
    t0 = time.time()
    if p.dim == 1:
        h, times, roughness = run_1d(p)
        if p.verbose:
            print(f"Final roughness W={roughness[-1]:.4f}")
        # Simple plot summary
        plt.figure(); plt.plot(times, roughness); plt.xlabel('t'); plt.ylabel('W'); plt.title('1D Roughness'); plt.tight_layout(); plt.show()
    else:
        collect_frames = p.save_anim
        h, times, roughness, frames = run_2d(p, collect_frames=collect_frames)
        if p.verbose:
            print(f"Final roughness W={roughness[-1]:.4f}")
        plt.figure(); plt.plot(times, roughness); plt.xlabel('t'); plt.ylabel('W'); plt.title('2D Roughness'); plt.tight_layout(); plt.show()
        if p.save_anim:
            save_animation_2d(frames, p)
    if p.verbose:
        print(f"Total runtime: {time.time()-t0:.2f}s")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")

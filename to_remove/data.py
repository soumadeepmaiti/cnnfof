from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

RADIUS_CONFIG: dict = {
    'r200b': {
        'file_col'   : 'r200b',   # directly in ROCKSTAR halos file [kpc/h]
        'mass_col'   : 'm200b',
        'overdensity': 200,
        'reference'  : 'b',
        'out_radius' : 'HR200b',
        'out_mass'   : 'HM200b',
    },
    'rvir': {
        'file_col'   : None,      # not a direct column → computed from mvir
        'mass_col'   : 'mvir',
        'overdensity': None,      # virial: Δ from Bryan & Norman 1998
        'reference'  : 'c',
        'out_radius' : 'HRvir',
        'out_mass'   : 'HMvir',
    },
    'r200c': {
        'file_col'   : None,
        'mass_col'   : 'm200c',
        'overdensity': 200,
        'reference'  : 'c',
        'out_radius' : 'HR200c',
        'out_mass'   : 'HM200c',
    },
    'r500c': {
        'file_col'   : None,
        'mass_col'   : 'm500c',
        'overdensity': 500,
        'reference'  : 'c',
        'out_radius' : 'HR500c',
        'out_mass'   : 'HM500c',
    },
}

# ROCKSTAR Halo_Details.ascii column names (particle–halo assignment table)
_HALO_DETAILS_COLS = [
    'Particle_ID', 'Px', 'Py', 'Pz', 'PVx', 'PVy', 'PVz',
    'Halo_ID', 'HaloType', 'Hx', 'Hy', 'Hz', 'HVx', 'HVy', 'HVz',
]

# ROCKSTAR halos_0.0.ascii column names (halo property table)
_HALOS_COLS = [
    'id', 'num_p', 'm200b', 'mbound_200b', 'r200b', 'vmax', 'rvmax', 'vrms',
    'x', 'y', 'z', 'vx', 'vy', 'vz', 'Jx', 'Jy', 'Jz', 'E', 'Spin',
    'PosUncertainty', 'VelUncertainty', 'bulk_vx', 'bulk_vy', 'bulk_vz',
    'BulkVelUnc', 'n_core', 'mvir', 'm200c', 'm500c', 'm1000c',
    'Xoff', 'Voff', 'spin_bullock', 'b_to_a', 'c_to_a',
    'A[x]', 'A[y]', 'A[z]', 'b_to_a(500c)', 'c_to_a(500c)',
    'A[x](500c)', 'A[y](500c)', 'A[z](500c)',
    'Rs', 'Rs_Klypin', 'T/|U|', 'M_pe_Behroozi', 'M_pe_Diemer',
    'Halfmass_Radius', 'idx', 'i_so', 'i_ph', 'num_cp', 'mmetric',
]


def _virial_overdensity(omega_m: float, z: float = 0.0) -> float:
    
    omega_lambda = 1.0 - omega_m
    E2 = omega_m * (1 + z) ** 3 + omega_lambda
    omega_mz = omega_m * (1 + z) ** 3 / E2   # Ω_m(z)
    x = omega_mz - 1.0
    return 18.0 * np.pi ** 2 + 82.0 * x - 39.0 * x ** 2


def _compute_so_radius(mass_msun_h: pd.Series,
                       overdensity: float,
                       reference: str,
                       omega_m: float,
                       z: float = 0.0) -> pd.Series:
    
    RHO_CRIT_0 = 2.775e11   # (Msun/h) / (Mpc/h)^3 at z=0

    omega_lambda = 1.0 - omega_m
    E2 = omega_m * (1 + z) ** 3 + omega_lambda   # E(z)^2
    rho_crit = RHO_CRIT_0 * E2                   # critical density at redshift z

    if reference == 'c':
        rho_ref = rho_crit
    elif reference == 'b':
        omega_mz = omega_m * (1 + z) ** 3 / E2
        rho_ref  = omega_mz * rho_crit
    else:
        raise ValueError(f"reference must be 'c' or 'b', got '{reference}'")

    # R [Mpc/h] = (3M / (4π Δ ρ_ref))^(1/3)
    radius_mpch = (3.0 * mass_msun_h / (4.0 * np.pi * overdensity * rho_ref)) ** (1.0 / 3.0)
    return radius_mpch * 1000.0   # Mpc/h → kpc/h



def load_halo_catalogue(halos_path: str,
                        radius_def: str,
                        h: float,
                        omega_m: float = 0.3186,
                        z: float = 0.0) -> pd.DataFrame:
    
    if radius_def not in RADIUS_CONFIG:
        raise ValueError(
            f"Unknown radius_def '{radius_def}'. "
            f"Choose from: {list(RADIUS_CONFIG.keys())}"
        )
    cfg      = RADIUS_CONFIG[radius_def]
    mass_col = cfg['mass_col']

    # Columns we need from the halos file
    need_cols = ['id', 'x', 'y', 'z', 'vx', 'vy', 'vz', mass_col]
    if cfg['file_col'] is not None:
        need_cols.append(cfg['file_col'])

    df = pd.read_csv(
        halos_path,
        sep      = r'\s+',
        comment  = '#',
        names    = _HALOS_COLS,
        skiprows = 1,
        usecols  = need_cols,
    )

    # Positions: Mpc/h → Mpc
    for c in ['x', 'y', 'z']:
        df[c] /= h

    # Radius → Mpc
    if cfg['file_col'] is not None:
        # directly available in kpc/h
        df[cfg['out_radius']] = df[cfg['file_col']] / (1000.0 * h)
    else:
        # compute from mass
        if cfg['overdensity'] is None:
            # virial: Δ from Bryan & Norman
            delta = _virial_overdensity(omega_m, z)
        else:
            delta = cfg['overdensity']

        radius_kpch = _compute_so_radius(
            df[mass_col], delta, cfg['reference'], omega_m, z
        )
        df[cfg['out_radius']] = radius_kpch / (1000.0 * h)   # kpc/h → Mpc

    # Rename for clarity
    df = df.rename(columns={
        'id'      : 'Halo_ID',
        'x'       : 'Hx',
        'y'       : 'Hy',
        'z'       : 'Hz',
        'vx'      : 'HVx',
        'vy'      : 'HVy',
        'vz'      : 'HVz',
        mass_col  : cfg['out_mass'],
    })

    keep = ['Halo_ID', 'Hx', 'Hy', 'Hz', 'HVx', 'HVy', 'HVz',
            cfg['out_radius'], cfg['out_mass']]
    return df[keep].copy()


def load_halo_particle_table(details_path: str, h: float) -> pd.DataFrame:
    df = pd.read_csv(
        details_path,
        sep     = r'\s+',
        comment = '#',
        names   = _HALO_DETAILS_COLS,
        skiprows= 1,
    )

    # Positions: Mpc/h → Mpc  (velocities already in km/s, no h conversion)
    for c in ['Px', 'Py', 'Pz']:
        df[c] /= h

    df['Particle_ID'] = df['Particle_ID'].astype(np.int64)
    df['Halo_ID']     = df['Halo_ID'].astype(np.int64)

    # Drop redundant halo-center columns — will be re-added from halos catalogue
    return df[['Particle_ID', 'Px', 'Py', 'Pz',
               'PVx', 'PVy', 'PVz', 'Halo_ID', 'HaloType']].copy()


def build_halo_membership(halo_particles: pd.DataFrame,
                          halos: pd.DataFrame,
                          radius_def: str,
                          min_particles: int = 25) -> pd.DataFrame:
    cfg = RADIUS_CONFIG[radius_def]
    radius_col = cfg['out_radius']
    mass_col   = cfg['out_mass']

    # Join particle table with halo properties
    df = halo_particles.merge(halos, on='Halo_ID', how='inner')

    # Euclidean distance particle → halo centre [Mpc]
    df['d_HP'] = np.sqrt(
        (df['Px'] - df['Hx']) ** 2 +
        (df['Py'] - df['Hy']) ** 2 +
        (df['Pz'] - df['Hz']) ** 2
    )

    # Normalised distance
    df['r_HP'] = df['d_HP'] / df[radius_col]

    # Keep only particles within the chosen radius
    df = df[df['d_HP'] <= df[radius_col]].copy()

    # Count particles per halo after radius filter
    counts = (
        df.groupby('Halo_ID')['Particle_ID']
          .count()
          .rename('Hpart')
          .reset_index()
    )

    # Discard under-resolved halos
    valid = counts[counts['Hpart'] >= min_particles]
    df = df.merge(valid[['Halo_ID', 'Hpart']], on='Halo_ID', how='inner')

    return df.reset_index(drop=True)


def build_training_sample(snap: dict,
                          halo_membership: pd.DataFrame,
                          radius_def: str) -> pd.DataFrame:
   
    cfg = RADIUS_CONFIG[radius_def]

    # Full snapshot as DataFrame
    snapshot_df = pd.DataFrame({
        'Particle_ID': snap['ids'],
        'Px'         : snap['pos'][:, 0],
        'Py'         : snap['pos'][:, 1],
        'Pz'         : snap['pos'][:, 2],
        'PVx'        : snap['vel'][:, 0],
        'PVy'        : snap['vel'][:, 1],
        'PVz'        : snap['vel'][:, 2],
    })

    # Halo membership columns to carry over
    halo_cols = [
        'Particle_ID', 'Halo_ID', 'HaloType',
        'Hx', 'Hy', 'Hz', 'HVx', 'HVy', 'HVz',
        cfg['out_radius'], cfg['out_mass'], 'Hpart', 'd_HP', 'r_HP',
    ]
    halo_df = halo_membership[halo_cols].copy()

    # Left join: every particle gets halo info where available, NaN otherwise
    combined = snapshot_df.merge(halo_df, on='Particle_ID', how='left')

    # Host flag: 1 = inside a halo, 0 = field
    combined['Host'] = combined['Halo_ID'].notna().astype(np.int8)

    # Canonical column order
    output_cols = [
        'Particle_ID',
        'Px', 'Py', 'Pz',
        'PVx', 'PVy', 'PVz',
        'Host',
        'Halo_ID', 'HaloType',
        'Hx', 'Hy', 'Hz',
        'HVx', 'HVy', 'HVz',
        cfg['out_radius'], cfg['out_mass'],
        'Hpart', 'd_HP', 'r_HP',
    ]
    for col in output_cols:
        if col not in combined.columns:
            combined[col] = np.nan

    combined = combined[output_cols].sort_values('Particle_ID').reset_index(drop=True)
    return combined


def save_dataset(df: pd.DataFrame,
                 output_base: str,
                 run_idx: int,
                 save_ascii: bool = True,
                 save_bin: bool   = True) -> dict:
   
    tag   = f"{run_idx:04d}"
    paths = {}

    if save_ascii:
        out_dir = os.path.join(output_base, 'ascii')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{tag}.ascii")
        df.to_csv(path, sep=' ', index=False, na_rep='NaN', float_format='%.6e')
        paths['ascii'] = path

    if save_bin:
        out_dir = os.path.join(output_base, 'bin')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{tag}.bin")
        df.to_pickle(path)
        paths['bin'] = path

    return paths


def process_run(run_idx      : int,
                snap         : dict,
                rockstar_base: str,
                output_base  : str,
                radius_def   : str   = 'r200b',
                h            : float = 0.67,
                min_particles: int   = 25,
                save_ascii   : bool  = True,
                save_bin     : bool  = True,
                omega_m      : float = 0.3186,
                z            : float = 0.0) -> pd.DataFrame | None:
   
    run_str      = f"{run_idx:04d}"
    halos_path   = os.path.join(rockstar_base + run_str, 'halos_0.0.ascii')
    details_path = os.path.join(rockstar_base + run_str, 'Halo_Details.ascii')

    missing = [p for p in [halos_path, details_path] if not os.path.exists(p)]
    if missing:
        print(f"[{run_str}] Skipping -- missing: {missing}")
        return None

    try:
        # 1. Load ROCKSTAR halo catalogue (positions + radius + mass)
        halos = load_halo_catalogue(halos_path, radius_def, h, omega_m, z)

        # 2. Load ROCKSTAR particle-halo assignment table
        halo_particles = load_halo_particle_table(details_path, h)

        # 3. Filter to particles within the chosen radius, drop small halos
        membership = build_halo_membership(
            halo_particles, halos, radius_def, min_particles
        )

        # 4. Merge with full snapshot -> labelled training sample
        df = build_training_sample(snap, membership, radius_def)

        # 5. Save
        paths = save_dataset(df, output_base, run_idx, save_ascii, save_bin)

        n_halo  = int(df['Host'].sum())
        n_field = len(df) - n_halo
        n_halos = int(df['Halo_ID'].nunique()) - 1
        print(f"[{run_str}] done  |  "
              f"{len(df):,} particles  "
              f"(halo {n_halo:,} / field {n_field:,})  "
              f"{n_halos} halos  "
              f"-> {list(paths.values())}")
        return df

    except Exception as exc:
        print(f"[{run_str}] ERROR: {exc}")
        return None


def process_all_runs(n_runs          : int,
                     snapshot_reader,
                     snapshot_base   : str,
                     rockstar_base   : str,
                     output_base     : str,
                     radius_def      : str   = 'r200b',
                     h               : float = 0.67,
                     min_particles   : int   = 25,
                     save_ascii      : bool  = True,
                     save_bin        : bool  = True,
                     omega_m         : float = 0.3186,
                     z               : float = 0.0,
                     run_range       : range | None = None) -> None:
   
    indices = run_range if run_range is not None else range(n_runs)

    print(f"Starting data prep  |  radius_def={radius_def}  "
          f"min_particles={min_particles}  runs={len(list(indices))}")
    print(f"  snapshot -> {snapshot_base}<run>/snapshot_000")
    print(f"  rockstar -> {rockstar_base}<run>/")
    print(f"  output   -> {output_base}/")
    print()

    t0 = time.perf_counter()
    success = 0
    skipped = 0

    for i in indices:
        run_str   = f"{i:04d}"
        snap_path = os.path.join(snapshot_base + run_str, 'snapshot_000')

        if not os.path.exists(snap_path) and not os.path.exists(snap_path + '.0'):
            print(f"[{run_str}] Skipping -- snapshot not found: {snap_path}")
            skipped += 1
            continue

        snap   = snapshot_reader(snap_path, h=h)
        result = process_run(
            run_idx       = i,
            snap          = snap,
            rockstar_base = rockstar_base,
            output_base   = output_base,
            radius_def    = radius_def,
            h             = h,
            min_particles = min_particles,
            save_ascii    = save_ascii,
            save_bin      = save_bin,
            omega_m       = omega_m,
            z             = z,
        )
        if result is not None:
            success += 1
        else:
            skipped += 1

    elapsed = time.perf_counter() - t0
    print()
    print(f"Finished  |  {success} runs OK  |  {skipped} skipped  "
          f"|  {elapsed:.1f}s  ({elapsed/max(success,1):.1f}s/run)")
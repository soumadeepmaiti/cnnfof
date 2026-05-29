"""
gadget4_reader.py
=================
Standalone GADGET-4 snapshot reader and writer.
Drop this file into your repo — no other dependencies needed beyond NumPy.

Notebook usage
--------------
    import gadget4_reader as g4

    snap = g4.read_snapshot("/path/to/snapshot_000", h=0.6774)

    snap["pos"]     # (N, 3)  float32  Mpc      — comoving positions
    snap["vel"]     # (N, 3)  float32  km/s     — physical peculiar velocities
    snap["ids"]     # (N,)    int64             — particle IDs
    snap["header"]  # dict    — BoxSize, redshift, scale factor, …

    # save to binary   → /output/cnnfof/bin/snapshot_000.npz
    g4.save_snapshot(snap, "/output/cnnfof", fmt="bin")

    # save to ascii    → /output/cnnfof/ascii/snapshot_000.txt
    g4.save_snapshot(snap, "/output/cnnfof", fmt="ascii")

Input units (GADGET-4 file)
----------------------------
    positions   Mpc/h   (comoving)
    velocities  km/s    stored as  v_pec / sqrt(a)  — corrected automatically

Output units (what you get back)
----------------------------------
    positions   Mpc     (comoving)   =  pos_Mpc_h / h
    velocities  km/s    (physical peculiar)  =  v_stored * sqrt(a)
    IDs         int64
"""

from __future__ import annotations

import array   as _carray
import os
import struct
import textwrap

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# GADGET-4 header memory map
# ─────────────────────────────────────────────────────────────────────────────
#
#  bytes   type      field
#  0–7     uint64    npart[0]        type-0 (gas) particles in this file
#  8–15    uint64    npart[1]        type-1 (DM)  particles in this file
#  16–23   uint64    npartTotal[0]   type-0 total across all sub-files
#  24–31   uint64    npartTotal[1]   type-1 total across all sub-files
#  32–39   float64   mass[0]         type-0 particle mass
#  40–47   float64   mass[1]         type-1 DM particle mass
#  48–55   float64   time            scale factor  a = 1 / (1 + z)
#  56–63   float64   redshift        z
#  64–71   float64   BoxSize         box side  [Mpc/h]
#  72–75   int32     num_files
#  76–79   int32     (padding)
#  80–87   int64     Ntrees
#  88–95   int64     Ntreestotal
#  ─────   ────────  ─────────────────────────────────────────────────
#  total   96 bytes

_HDR_FMT  = "=QQQQdddddiiqq"
_HDR_SIZE = struct.calcsize(_HDR_FMT)   # 96


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def read_snapshot(snapshot_path: str, h: float) -> dict:
    """Read a GADGET-4 DM snapshot.  Positions are returned in Mpc.

    Parameters
    ----------
    snapshot_path : str
        Path to the snapshot root, e.g. ``"/runs/sim/snapshot_000"``.
        If the bare path does not exist, ``<path>.0`` is tried automatically
        for multi-file snapshots.
    h : float
        Hubble parameter (dimensionless), e.g. ``0.6774``.
        Used to convert positions from Mpc/h  →  Mpc.

    Returns
    -------
    dict
        pos    : ndarray  (N, 3)  float32   comoving positions  [Mpc]
        vel    : ndarray  (N, 3)  float32   physical peculiar velocity  [km/s]
        ids    : ndarray  (N,)    int64     particle IDs
        header : dict
            BoxSize_mpch   box side length  [Mpc/h]
            BoxSize_mpc    box side length  [Mpc]   = BoxSize_mpch / h
            Redshift       output redshift  z
            ScaleFactor    scale factor  a = 1/(1+z)
            NpartTotal     total DM particle count
            ParticleMass   DM particle mass  [GADGET internal mass units]
            NumFiles       number of sub-files
            h              the h value you passed in
    """
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")

    header  = _read_header(snapshot_path)
    N       = header["NpartTotal"]
    a       = header["ScaleFactor"]
    nfiles  = header["NumFiles"]

    pos = np.empty((N, 3), dtype=np.float32)
    vel = np.empty((N, 3), dtype=np.float32)
    ids = np.empty(N,       dtype=np.int64)

    cursor = 0
    for ifile in range(nfiles):
        fpath  = _resolve(snapshot_path, ifile, nfiles)
        cursor = _read_file(fpath, pos, vel, ids, cursor)

    # ── Unit conversions ──────────────────────────────────────────────────
    pos /= np.float32(h)                  # Mpc/h  →  Mpc
    vel *= np.float32(np.sqrt(a))         # v/√a   →  physical km/s

    header["BoxSize_mpc"]   = header["BoxSize_mpch"] / h
    header["h"]             = h
    header["snapshot_name"] = os.path.basename(snapshot_path)  # e.g. "snapshot_000"

    return {"pos": pos, "vel": vel, "ids": ids, "header": header}


def save_snapshot(snap: dict,
                  output_dir: str,
                  fmt: str = "bin") -> str:
    """Save a snapshot dict to disk.

    The output directory is created automatically.  A subfolder ``bin/`` or
    ``ascii/`` is created inside ``output_dir`` depending on ``fmt``, and the
    file is named after the original snapshot (stored in ``snap["header"]``).

    Parameters
    ----------
    snap : dict
        Dict returned by :func:`read_snapshot`.
    output_dir : str
        Base directory where the output will be saved.
        ``fmt="bin"``   writes to  ``<output_dir>/bin/<name>.npz``
        ``fmt="ascii"`` writes to  ``<output_dir>/ascii/<name>.txt``
    fmt : {"bin", "ascii"}
        ``"bin"``   — NumPy compressed binary (``.npz``).  Fast, compact, exact.
        ``"ascii"`` — Space-delimited plain text (``.txt``).
                      Columns:  ``id  x[Mpc]  y[Mpc]  z[Mpc]  vx  vy  vz``

    Returns
    -------
    str
        Full path of the file that was written.

    Examples
    --------
    >>> g4.save_snapshot(snap, "/output/cnnfof", fmt="bin")
    '/output/cnnfof/bin/snapshot_000.npz'

    >>> g4.save_snapshot(snap, "/output/cnnfof", fmt="ascii")
    '/output/cnnfof/ascii/snapshot_000.txt'
    """
    fmt = fmt.lower().strip()
    if fmt not in ("bin", "ascii"):
        raise ValueError(f"fmt must be 'bin' or 'ascii', got '{fmt}'")

    pos    = snap["pos"]
    vel    = snap["vel"]
    ids    = snap["ids"]
    header = snap["header"]
    name   = header.get("snapshot_name", "snapshot")

    if fmt == "bin":
        save_dir = os.path.join(output_dir, "bin")
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(save_dir, name + ".npz")
        np.savez_compressed(
            out_file,
            pos    = pos,
            vel    = vel,
            ids    = ids,
            # header scalars stored as 0-d arrays so they round-trip cleanly
            BoxSize_mpc  = np.float64(header["BoxSize_mpc"]),
            BoxSize_mpch = np.float64(header["BoxSize_mpch"]),
            Redshift     = np.float64(header["Redshift"]),
            ScaleFactor  = np.float64(header["ScaleFactor"]),
            NpartTotal   = np.int64(header["NpartTotal"]),
            ParticleMass = np.float64(header["ParticleMass"]),
            h            = np.float64(header["h"]),
        )

    else:   # ascii
        save_dir = os.path.join(output_dir, "ascii")
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(save_dir, name + ".txt")
        header_lines = textwrap.dedent(f"""\
            # GADGET-4 snapshot — exported by gadget4_reader.py
            # Redshift        : {header['Redshift']:.6f}
            # ScaleFactor     : {header['ScaleFactor']:.6f}
            # BoxSize_mpc     : {header['BoxSize_mpc']:.6f}  Mpc
            # BoxSize_mpch    : {header['BoxSize_mpch']:.6f}  Mpc/h
            # h               : {header['h']}
            # NpartTotal      : {header['NpartTotal']}
            # ParticleMass    : {header['ParticleMass']}
            # Columns         : id  x[Mpc]  y[Mpc]  z[Mpc]  vx[km/s]  vy[km/s]  vz[km/s]
        """)
        data = np.column_stack([
            ids.astype(np.float64),
            pos.astype(np.float64),
            vel.astype(np.float64),
        ])
        with open(out_file, "w") as fout:
            fout.write(header_lines)
            np.savetxt(fout, data, fmt="%d %.6f %.6f %.6f %.4f %.4f %.4f")

    print(f"Saved {snap['ids'].shape[0]:,} particles  →  {out_file}")
    return out_file


def load_saved(path: str) -> dict:
    """Reload a snapshot previously saved in ``fmt='bin'`` (.npz) format.

    Parameters
    ----------
    path : str
        Path with or without the ``.npz`` extension.

    Returns
    -------
    dict
        Same structure as :func:`read_snapshot` output.
    """
    if not path.endswith(".npz"):
        path = path + ".npz"
    d = np.load(path)
    header = {
        "BoxSize_mpc" : float(d["BoxSize_mpc"]),
        "BoxSize_mpch": float(d["BoxSize_mpch"]),
        "Redshift"    : float(d["Redshift"]),
        "ScaleFactor" : float(d["ScaleFactor"]),
        "NpartTotal"  : int(d["NpartTotal"]),
        "ParticleMass": float(d["ParticleMass"]),
        "h"           : float(d["h"]),
    }
    return {"pos": d["pos"], "vel": d["vel"], "ids": d["ids"], "header": header}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_header(snapshot_path: str) -> dict:
    path = _resolve(snapshot_path, 0, nfiles=None)
    with open(path, "rb") as fd:
        bs = _read4(fd)
        if bs != _HDR_SIZE:
            raise ValueError(
                f"Header block is {bs} bytes; expected {_HDR_SIZE}. "
                f"Is this a GADGET-4 binary snapshot?  File: {path}"
            )
        raw = fd.read(_HDR_SIZE)
    (_, npart1, _, nptot1,
     _, mass1, time, redshift, BoxSize,
     num_files, _pad, Ntrees, Ntreestotal) = struct.unpack(_HDR_FMT, raw)
    return {
        "BoxSize_mpch": BoxSize,
        "Redshift"    : redshift,
        "ScaleFactor" : time,
        "NpartTotal"  : int(nptot1),
        "NpartHere"   : int(npart1),
        "ParticleMass": mass1,
        "NumFiles"    : num_files,
    }


def _read_file(fpath: str,
               pos: np.ndarray,
               vel: np.ndarray,
               ids: np.ndarray,
               cursor: int) -> int:
    """Read one snapshot sub-file; fill slices of pos/vel/ids from cursor."""
    with open(fpath, "rb") as fin:

        # header (re-read to get local particle count for this sub-file)
        s1 = _read4(fin);  raw = fin.read(_HDR_SIZE);  s2 = _read4(fin)
        _guard(s1, s2, "header", fpath)
        nhere = int(struct.unpack(_HDR_FMT, raw)[1])
        end   = cursor + nhere

        # positions  —  N × 3 × float32
        s1 = _read4(fin)
        _size_guard(s1, nhere, 3, 4, "positions", fpath)
        buf = _carray.array("f");  buf.fromfile(fin, nhere * 3)
        _guard(_read4(fin), s1, "positions", fpath)
        pos[cursor:end] = np.reshape(buf, (nhere, 3))

        # velocities  —  N × 3 × float32
        s1 = _read4(fin)
        _size_guard(s1, nhere, 3, 4, "velocities", fpath)
        buf = _carray.array("f");  buf.fromfile(fin, nhere * 3)
        _guard(_read4(fin), s1, "velocities", fpath)
        vel[cursor:end] = np.reshape(buf, (nhere, 3))

        # IDs  —  N × (4B uint32  or  8B uint64), auto-detected
        s1 = _read4(fin)
        if   nhere == s1 // 8:  ifmt = "Q"
        elif nhere == s1 // 4:  ifmt = "I"
        else:
            raise ValueError(
                f"Cannot detect ID width in {fpath}: "
                f"block={s1}B, npart={nhere}"
            )
        buf = _carray.array(ifmt);  buf.fromfile(fin, nhere)
        _guard(_read4(fin), s1, "IDs", fpath)
        ids[cursor:end] = np.asarray(buf, dtype=np.int64)

    return end


def _resolve(path: str, ifile: int, nfiles) -> str:
    """Return the actual file path for sub-file ifile."""
    # multi-file: always use the .N suffix
    if nfiles is not None and nfiles > 1:
        candidate = f"{path}.{ifile}"
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Sub-file not found: {candidate}")
        return candidate
    # single file: bare path first, then .0
    if os.path.exists(path):
        return path
    alt = path + ".0"
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(
        f"Snapshot not found: '{path}'  (also tried '{alt}')"
    )


def _read4(fd) -> int:
    return struct.unpack("I", fd.read(4))[0]


def _guard(s1, s2, name, path):
    if s1 != s2:
        raise IOError(
            f"Fortran record marker mismatch in '{name}' block of {path}: "
            f"prefix={s1}, suffix={s2}.  File may be corrupt."
        )


def _size_guard(block_bytes, n, ndim, itemsize, name, path):
    expected = n * ndim * itemsize
    if block_bytes != expected:
        raise IOError(
            f"'{name}' block size mismatch in {path}: "
            f"got {block_bytes}B, expected {expected}B "
            f"({n} particles × {ndim} × {itemsize}B)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI:  python gadget4_reader.py <snapshot_path> <h>  [--save fmt]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys, time

    ap = argparse.ArgumentParser(
        description="Read a GADGET-4 snapshot and print summary statistics."
    )
    ap.add_argument("snapshot_path", help="Snapshot root path")
    ap.add_argument("h",  type=float, help="Hubble parameter (e.g. 0.6774)")
    ap.add_argument("--save", metavar="FMT",
                    help="Optionally save output: 'bin' or 'ascii'")
    ap.add_argument("--out", metavar="PATH", default="snapshot_out",
                    help="Output path for --save (no extension, default: snapshot_out)")
    args = ap.parse_args()

    t0   = time.perf_counter()
    snap = read_snapshot(args.snapshot_path, h=args.h)
    dt   = time.perf_counter() - t0

    hdr = snap["header"]
    print(f"\n── Header ───────────────────────────────────────────")
    print(f"  Redshift       {hdr['Redshift']:.4f}")
    print(f"  Scale factor   {hdr['ScaleFactor']:.4f}")
    print(f"  BoxSize        {hdr['BoxSize_mpch']:.2f} Mpc/h  "
          f"=  {hdr['BoxSize_mpc']:.2f} Mpc")
    print(f"  Npart total    {hdr['NpartTotal']:,}")
    print(f"  Particle mass  {hdr['ParticleMass']:.4e}")
    print(f"\n── Data ─────────────────────────────────────────────")
    print(f"  pos  {snap['pos'].shape}  "
          f"[{snap['pos'].min():.3f}, {snap['pos'].max():.3f}] Mpc")
    print(f"  vel  {snap['vel'].shape}  "
          f"[{snap['vel'].min():.1f}, {snap['vel'].max():.1f}] km/s")
    print(f"  ids  {snap['ids'].shape}  "
          f"[{snap['ids'].min()}, {snap['ids'].max()}]")
    print(f"  Read in {dt*1e3:.1f} ms")

    if args.save:
        save_snapshot(snap, args.out, fmt=args.save)

    sys.exit(0)
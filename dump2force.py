#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump2force.py
LIGGGHTS dump (pair/local) -> VTK/VTU line network writer + Louvain attributes

主なポイント（今回の修正点を含む）
- VTU/VTK どちらも出力可能 (--vtk-format {vtu,vtk}, --encoding {ascii,binary}).
- PVD は出力しません（要求により削除）。
- 新規フォルダを作りません。デフォルト出力先は「入力ファイルと同じディレクトリ」です。
  （--outdir で明示した既存ディレクトリに出すことは可能ですが、自動作成はしません）
- VTK legacy (ASCII/BINARY) で ParaView が "Unsupported cell attribute type: nan" などを出す
  ことがあるため、VTK legacy 出力時は NaN/Inf を --nan-fill の値（既定 0.0）に置換します。

拡張仕様（従来どおり）
- "ITEM: ENTRIES ..." から列名を解釈（必須 12 列: x1 y1 z1 x2 y2 z2 id1 id2 periodic fx fy fz）。
- 13 列目以降の任意スカラー列、連続 3 列のベクトル列（*x/*y/*z / *1/*2/*3 / *[1]-[3]）をパススルー。
- Louvain 法（python-louvain）により無向グラフでコミュニティ検出：
  * ノード = 粒子 ID (id1/id2)
  * エッジ重み = 接触力の大きさ (fx,fy,fz のノルム)。<=0 は無視。
  * --seed, --resolution で制御（デフォルト 42, 1.0）。
- CellData（各線分）: force, connectionLength に加え、任意スカラー/ベクトル（3 成分）を転送。
  * 'community', 'intra_comm'（同一コミュニティ=1）
  * 'comm_mean_*' / 'comm_sum_*'（同一コミュニティ内部のエッジに対して平均/合計。境界エッジは NaN）
- PointData（--write-pointdata 指定時）:
  * node_community, node_degree（無重み次数）, node_force_sum（重み総和）

依存:
- numpy
- networkx
- python-louvain (import community as community_louvain)
"""

from __future__ import annotations

import argparse
import gzip
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional imports for Louvain
try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None  # type: ignore

try:
    from community import community_louvain  # type: ignore
except Exception:
    try:
        import community as community_louvain  # type: ignore
    except Exception:
        community_louvain = None  # type: ignore


# ============================================================
#                    Minimal bdump (Python 3) with headers
# ============================================================

class Snap:
    """One snapshot."""
    __slots__ = ("time", "natoms", "atoms", "colnames")
    def __init__(self):
        self.time: int = 0
        self.natoms: int = 0
        self.atoms: Optional[np.ndarray] = None  # shape: (natoms, ncols) float
        self.colnames: List[str] = []  # tokens from "ITEM: ENTRIES ..."


class BDump:
    """
    Very small subset of Pizza.py bdump functionality, adapted for Python 3,
    sufficient to read LIGGGHTS dump from compute pair/local style.  Supports .gz.

    Incremental iteration yields timesteps in file order.
    """

    def __init__(self, filename: str, read_all: bool = False):
        self.flist: List[str] = [filename]
        if not self.flist:
            raise RuntimeError("No dump file specified")

        self.snaps: List[Snap] = []
        self.nsnaps = 0

        self.increment = 0 if read_all else 1
        self.nextfile = 0
        self.eof = 0

        if read_all:
            self._read_all()

    def __iter__(self):
        return self

    def __next__(self):
        val = self.next()
        if val < 0:
            raise StopIteration
        return val

    def next(self) -> int:
        """Read next snapshot from the file list; return timestep or -1 if none."""
        if not self.increment:
            raise RuntimeError("BDump created in non-incremental mode")

        while True:
            fname = self.flist[self.nextfile]
            is_gz = fname.endswith(".gz")
            if is_gz:
                with gzip.open(fname, "rt") as f:
                    f.read(self.eof)
                    snap, consumed = self._read_snapshot_text(f)
                    if not snap:
                        self.nextfile += 1
                        if self.nextfile == len(self.flist):
                            return -1
                        self.eof = 0
                        continue
                    self.eof += consumed
            else:
                with open(fname, "rb") as fb:
                    fb.seek(self.eof)
                    snap = self._read_snapshot_binary_text(fb)
                    if not snap:
                        self.nextfile += 1
                        if self.nextfile == len(self.flist):
                            return -1
                        self.eof = 0
                        continue
                    self.eof = fb.tell()

            # de-duplicate timesteps (skip if already present)
            already = any(s.time == snap.time for s in self.snaps)
            if already:
                continue

            self.snaps.append(snap)
            self.nsnaps += 1
            return snap.time

    # --------------------------- helpers ---------------------------

    def _read_all(self):
        for fname in self.flist:
            opener = gzip.open if fname.endswith(".gz") else open
            mode = "rt" if fname.endswith(".gz") else "rb"
            with opener(fname, mode) as f:
                while True:
                    snap = (self._read_snapshot_text(f) if fname.endswith(".gz")
                            else self._read_snapshot_binary_text(f))
                    if not snap:
                        break
                    self.snaps.append(snap)
        # sort and deduplicate
        self.snaps.sort(key=lambda s: s.time)
        unique: List[Snap] = []
        for s in self.snaps:
            if not unique or s.time != unique[-1].time:
                unique.append(s)
        self.snaps = unique
        self.nsnaps = len(self.snaps)

    # --------------------------- readers ---------------------------

    def _read_snapshot_text(self, f_text) -> Tuple[Optional[Snap], int]:
        """
        gzip.open(...,'rt') などテキストモードのストリームから 1 ステップ読む。
        戻り値は (snap, 消費文字数)。BOX BOUNDS が挟まるケースに対応。
        """
        consumed = 0
        def _rline():
            nonlocal consumed
            s = f_text.readline()
            consumed += len(s)
            return s

        try:
            # ITEM: TIMESTEP
            line = _rline()
            if not line:
                return None, consumed

            # timestep
            line = _rline()
            if not line:
                return None, consumed
            t = int(line.strip().split()[0])

            # ITEM: NUMBER OF ENTRIES
            _rline()
            line = _rline()
            if not line:
                return None, consumed
            natoms = int(line.strip().split()[0])

            # 直後の行を覗く
            header = _rline()
            if not header:
                return None, consumed

            if header.startswith("ITEM: BOX BOUNDS"):
                # 境界3行
                _rline(); _rline(); _rline()
                header = _rline()
                if not header:
                    return None, consumed

            tokens = header.strip().split()
            try:
                idx = tokens.index("ENTRIES")
                colnames = tokens[idx + 1:]
            except ValueError:
                colnames = tokens[2:] if len(tokens) >= 2 else []

            snap = Snap()
            snap.time = t
            snap.natoms = natoms
            snap.colnames = colnames

            if natoms > 0:
                first = _rline().split()
                ncol = len(first)
                rows = [first]
                for _ in range(natoms - 1):
                    rows.append(_rline().split())
                data = np.array(rows, dtype=np.float64)
                if data.shape[1] != ncol:
                    return None, consumed
                snap.atoms = data
            else:
                snap.atoms = np.zeros((0, len(colnames)), dtype=np.float64)

            return snap, consumed
        except Exception:
            return None, consumed



    def _read_snapshot_binary_text(self, fb) -> Optional[Snap]:
        """
        非 gzip ファイルを 'rb' で開いたハンドルから 1 ステップ読む。
        NUMBER OF ENTRIES の直後に BOX BOUNDS が挟まるケースに対応。
        """
        try:
            # ITEM: TIMESTEP
            line = fb.readline()
            if not line:
                return None

            # timestep 値
            line = fb.readline()
            if not line:
                return None
            t = int(line.split()[0])

            # ITEM: NUMBER OF ENTRIES
            hdr = fb.readline()
            if not hdr:
                return None

            # N (= 行数)
            line = fb.readline()
            if not line:
                return None
            natoms = int(line.strip().split()[0])

            # 次の行を覗く
            header = fb.readline()
            if not header:
                return None

            try:
                header_str = header.decode("utf-8", errors="ignore").strip()
            except AttributeError:
                header_str = str(header).strip()

            # もし BOX BOUNDS なら3行読み飛ばし、次の ENTRIES 行を読む
            if header_str.startswith("ITEM: BOX BOUNDS"):
                # 境界3行
                fb.readline(); fb.readline(); fb.readline()
                header = fb.readline()
                if not header:
                    return None
                try:
                    header_str = header.decode("utf-8", errors="ignore").strip()
                except AttributeError:
                    header_str = str(header).strip()

            # "ITEM: ENTRIES x1 y1 ..." から列名を取る
            tokens = header_str.split()
            try:
                idx = tokens.index("ENTRIES")
                colnames = tokens[idx + 1:]
            except ValueError:
                colnames = tokens[2:] if len(tokens) >= 2 else []

            snap = Snap()
            snap.time = t
            snap.natoms = natoms
            snap.colnames = colnames

            # データ行
            if natoms > 0:
                first = fb.readline().split()
                ncol = len(first)
                rows = [first]
                for _ in range(natoms - 1):
                    rows.append(fb.readline().split())
                data = np.array(rows, dtype=np.float64)
                if data.shape[1] != ncol:
                    return None
                snap.atoms = data
            else:
                snap.atoms = np.zeros((0, len(colnames)), dtype=np.float64)

            return snap
        except Exception:
            return None



# ============================================================
#                 Force network processing
# ============================================================

REQUIRED12 = ["x1","y1","z1","x2","y2","z2","id1","id2","periodic","fx","fy","fz"]

def _parse_column_indices(colnames: List[str]) -> Dict[str, int]:
    """Map required names to indices; validate the 12 mandatory columns exist."""
    name_to_idx: Dict[str,int] = {}
    for req in REQUIRED12:
        try:
            name_to_idx[req] = colnames.index(req)
        except ValueError:
            raise RuntimeError(f"Required column '{req}' not found in ENTRIES header: {colnames}")
    return name_to_idx

def _split_optional_columns(colnames: List[str]) -> Tuple[List[str], Dict[str, Tuple[int,int,int]]]:
    """
    ENTRIES の 13 列目以降から、任意スカラーとベクトル(3 成分)を抽出。
    ベクトル認識：連続 3 列が (suffix = x/y/z) または (1/2/3) または ([1]/[2]/[3])。
    返り値: (scalar_names, vector_triplets{prefix->(ix,iy,iz)})
    """
    import re

    if len(colnames) <= 12:
        return [], {}

    start = 12
    n = len(colnames)

    def ends_xyz(name: str) -> Optional[str]:
        if not name:
            return None
        if name[-1].lower() in ("x","y","z"):
            return name[:-1]
        return None

    re_num = re.compile(r"^(?P<prefix>.+?)(?P<axis>[123])$")
    re_brk = re.compile(r"^(?P<prefix>.+?)\[(?P<idx>\d+)\]$")

    vector_triplets: Dict[str, Tuple[int,int,int]] = {}
    covered: set = set()

    i = start
    while i <= n - 3:
        a, b, c = colnames[i], colnames[i+1], colnames[i+2]

        # A) ...x/...y/...z
        pa, pb, pc = ends_xyz(a), ends_xyz(b), ends_xyz(c)
        if pa is not None and pb is not None and pc is not None:
            suf = (a[-1].lower(), b[-1].lower(), c[-1].lower())
            if set(suf) == {"x","y","z"}:
                prefix = pa
                if pb == prefix and pc == prefix:
                    ix = i + suf.index("x")
                    iy = i + suf.index("y")
                    iz = i + suf.index("z")
                    vector_triplets[prefix] = (ix, iy, iz)
                    covered.update({colnames[ix], colnames[iy], colnames[iz]})
                    i += 3
                    continue

        # B) ...1/...2/...3
        mb = re_num.match(a); nb = re_num.match(b); cb = re_num.match(c)
        if mb and nb and cb:
            pre_a, ax_a = mb.group("prefix"), mb.group("axis")
            pre_b, ax_b = nb.group("prefix"), nb.group("axis")
            pre_c, ax_c = cb.group("prefix"), cb.group("axis")
            if pre_a == pre_b == pre_c and {ax_a, ax_b, ax_c} == {"1","2","3"}:
                trip = {ax_a: i, ax_b: i+1, ax_c: i+2}
                ix, iy, iz = trip["1"], trip["2"], trip["3"]
                vector_triplets[pre_a] = (ix, iy, iz)
                covered.update({colnames[ix], colnames[iy], colnames[iz]})
                i += 3
                continue

        # C) ...[1]/...[2]/...[3]
        ma = re_brk.match(a); nb2 = re_brk.match(b); cc2 = re_brk.match(c)
        if ma and nb2 and cc2:
            pre_a, ia = ma.group("prefix"), ma.group("idx")
            pre_b, ib = nb2.group("prefix"), nb2.group("idx")
            pre_c, ic = cc2.group("prefix"), cc2.group("idx")
            if pre_a == pre_b == pre_c and {ia, ib, ic} == {"1","2","3"}:
                trip = {ia: i, ib: i+1, ic: i+2}
                ix, iy, iz = trip["1"], trip["2"], trip["3"]
                vector_triplets[pre_a] = (ix, iy, iz)
                covered.update({colnames[ix], colnames[iy], colnames[iz]})
                i += 3
                continue

        i += 1

    scalar_names = []
    for k in range(start, n):
        name = colnames[k]
        if name not in covered:
            scalar_names.append(name)

    return scalar_names, vector_triplets



def _build_geometry_and_base_celldata(snap: Snap,
                                      keep_periodic: bool) -> Tuple[np.ndarray, np.ndarray, Dict[str,np.ndarray], Dict[str,np.ndarray], np.ndarray, Tuple[np.ndarray,np.ndarray], List[str], Dict[str,Tuple[int,int,int]]]:
    """Convert a snapshot to geometry + base CellData (force, connectionLength) + optional pass-throughs."""
    atoms = snap.atoms
    if atoms is None:
        atoms = np.zeros((0,0), dtype=np.float64)
    colnames = snap.colnames
    if len(colnames) < 12:
        raise RuntimeError("Dump requires at least 12 columns (x1 y1 z1 x2 y2 z2 id1 id2 periodic fx fy fz)." )

    idx = _parse_column_indices(colnames)
    scalar_extras, vector_triplets = _split_optional_columns(colnames)

    # periodic mask
    periodic = atoms[:, idx["periodic"]].astype(np.int64) != 0
    mask = ~periodic if not keep_periodic else np.ones(atoms.shape[0], dtype=bool)

    # extract id arrays
    id1_all = atoms[:, idx["id1"]].astype(np.int64)
    id2_all = atoms[:, idx["id2"]].astype(np.int64)
    id1 = id1_all[mask]
    id2 = id2_all[mask]
    nconn = id1.shape[0]

    # unique particle IDs among kept contacts
    ids = np.unique(np.concatenate([id1, id2]))
    npts = ids.size
    id_to_idx = {pid: i for i, pid in enumerate(ids)}

    # gather positions for each unique id (pick from id1 row if available else id2)
    points = np.zeros((npts, 3), dtype=np.float64)
    first_row_by_id1: Dict[int,int] = {}
    first_row_by_id2: Dict[int,int] = {}
    for i,pid in enumerate(id1_all):
        if pid not in first_row_by_id1:
            first_row_by_id1[int(pid)] = i
    for i,pid in enumerate(id2_all):
        if pid not in first_row_by_id2:
            first_row_by_id2[int(pid)] = i

    for pid, pidx in id_to_idx.items():
        if pid in first_row_by_id1:
            j = first_row_by_id1[pid]
            points[pidx,0] = atoms[j, idx["x1"]]
            points[pidx,1] = atoms[j, idx["y1"]]
            points[pidx,2] = atoms[j, idx["z1"]]
        else:
            j = first_row_by_id2[pid]
            points[pidx,0] = atoms[j, idx["x2"]]
            points[pidx,1] = atoms[j, idx["y2"]]
            points[pidx,2] = atoms[j, idx["z2"]]

    # connectivity (point indices)
    lines = np.empty((nconn, 2), dtype=np.int32)
    for k in range(nconn):
        lines[k,0] = id_to_idx[int(id1[k])]
        lines[k,1] = id_to_idx[int(id2[k])]

    # base cell data（接続長と力の大きさ）
    dx = atoms[:, idx["x1"]] - atoms[:, idx["x2"]]
    dy = atoms[:, idx["y1"]] - atoms[:, idx["y2"]]
    dz = atoms[:, idx["z1"]] - atoms[:, idx["z2"]]
    connection_len = np.sqrt(dx*dx + dy*dy + dz*dz)[mask].astype(np.float64, copy=False)

    fx = atoms[:, idx["fx"]]; fy = atoms[:, idx["fy"]]; fz = atoms[:, idx["fz"]]
    fmag = np.sqrt(fx*fx + fy*fy + fz*fz)[mask].astype(np.float64, copy=False)

    cell_scalars: Dict[str,np.ndarray] = {
        "force": fmag,
        "connectionLength": connection_len,
    }

    # extra scalar pass-throughs
    for name in scalar_extras:
        col = atoms[:, colnames.index(name)][mask].astype(np.float64, copy=False)
        cell_scalars[name] = col

    # extra vector pass-throughs
    cell_vectors: Dict[str,np.ndarray] = {}
    for prefix, (ix,iy,iz) in vector_triplets.items():
        vx = atoms[:, ix][mask].astype(np.float64, copy=False)
        vy = atoms[:, iy][mask].astype(np.float64, copy=False)
        vz = atoms[:, iz][mask].astype(np.float64, copy=False)
        vec = np.stack([vx,vy,vz], axis=1)  # (M,3)
        cell_vectors[prefix] = vec

    return points, lines, cell_scalars, cell_vectors, ids.astype(np.int64), (id1, id2), scalar_extras, vector_triplets


def _annotate_louvain(ids: np.ndarray,
                      id_pairs: Tuple[np.ndarray,np.ndarray],
                      weight_for_graph: np.ndarray,
                      resolution: float,
                      seed: int,
                      write_pointdata: bool,
                      comm_targets: Dict[str, np.ndarray]
                      ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Louvain によるコミュニティ付与（エッジ/ノード属性を返す）。"""
    if community_louvain is None or nx is None:
        raise SystemExit("Louvain requires 'python-louvain' and 'networkx'. Install: pip install python-louvain networkx")

    id1, id2 = id_pairs
    M = weight_for_graph.shape[0]

    # Build graph: positive weights only
    G = nx.Graph()
    G.add_nodes_from([int(x) for x in ids.tolist()])
    for a, b, w in zip(id1.tolist(), id2.tolist(), weight_for_graph.tolist()):
        try:
            wv = float(w)
        except Exception:
            continue
        if wv > 0.0 and a != b:
            if G.has_edge(int(a), int(b)):
                G[int(a)][int(b)]["weight"] += wv
            else:
                G.add_edge(int(a), int(b), weight=wv)

    # partition
    part = community_louvain.best_partition(G, weight="weight", resolution=resolution, random_state=seed)

    # Arrays aligned to 'ids' order
    node_community = np.array([int(part.get(int(pid), -1)) for pid in ids], dtype=np.int64)

    # edge-level community & intra flag
    comm_map = {int(pid): int(cid) for pid, cid in part.items()}
    comm_e = np.empty(M, dtype=np.int64)
    intra = np.zeros(M, dtype=np.int32)
    for k in range(M):
        ca = comm_map.get(int(id1[k]), -1)
        cb = comm_map.get(int(id2[k]), -1)
        if ca == cb and ca != -1:
            comm_e[k] = ca
            intra[k] = 1
        else:
            comm_e[k] = -1
            intra[k] = 0

    # community-wise aggregations for every target
    from collections import defaultdict
    edges_by_comm: Dict[int, List[int]] = defaultdict(list)
    for idx_e, cid in enumerate(comm_e):
        if cid != -1:
            edges_by_comm[int(cid)].append(idx_e)

    cell_ann: Dict[str, np.ndarray] = {
        "community": comm_e,
        "intra_comm": intra.astype(np.float64),  # VTK 一貫性のため float
    }

    for name, arr in comm_targets.items():
        arr = np.asarray(arr, dtype=np.float64)
        out_mean = np.full(M, np.nan, dtype=np.float64)
        out_sum  = np.full(M, np.nan, dtype=np.float64)
        for cid, idxs in edges_by_comm.items():
            vals = arr[idxs]
            if len(vals):
                out_sum[idxs]  = float(np.nansum(vals))
                out_mean[idxs] = float(np.nanmean(vals))
        cell_ann[f"comm_mean_{name}"] = out_mean
        cell_ann[f"comm_sum_{name}"]  = out_sum

    degree_map = dict(G.degree())
    strength_map = dict(G.degree(weight="weight"))
    node_degree = np.array([int(degree_map.get(int(pid), 0)) for pid in ids], dtype=np.int64)
    node_force_sum = np.array([float(strength_map.get(int(pid), 0.0)) for pid in ids], dtype=np.float64)

    point_ann: Dict[str, np.ndarray] = {}
    if write_pointdata:
        point_ann = {
            "node_community": node_community.astype(np.float64),
            "node_degree": node_degree.astype(np.float64),
            "node_force_sum": node_force_sum.astype(np.float64),
        }

    return cell_ann, point_ann


# ============================================================
#                   VTK / VTU writers (scalars & vectors)
# ============================================================

def _sanitize_finite_scalar(a: np.ndarray, fill: float) -> np.ndarray:
    """Return float32 array with NaN/Inf replaced by 'fill'."""
    b = np.asarray(a, dtype=np.float32).copy(order="C")
    mask = ~np.isfinite(b)
    if mask.any():
        b[mask] = fill
    return b

def _sanitize_finite_vector(v: np.ndarray, fill: float) -> np.ndarray:
    """Return float32 (N,3) array with NaN/Inf replaced by 'fill'."""
    b = np.asarray(v, dtype=np.float32).reshape(-1,3).copy(order="C")
    mask = ~np.isfinite(b)
    if mask.any():
        b[mask] = fill
    return b


# ---------- VTU (XML) ----------

def _vtu_ascii_dataarrays(f, tag: str, scalars: Dict[str,np.ndarray], vectors: Dict[str,np.ndarray], nitems: int):
    """Write <PointData> or <CellData> ASCII arrays (Float64)."""
    f.write(f'      <{tag}>\n')
    for name, arr in scalars.items():
        f.write(f'        <DataArray type="Float64" Name="{name}" format="ascii">\n')
        if nitems:
            flat = " ".join(str(float(x)) for x in np.asarray(arr, dtype=np.float64).ravel(order="C"))
            f.write("          " + flat + "\n")
        f.write('        </DataArray>\n')
    for name, vec in vectors.items():
        f.write(f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="3" format="ascii">\n')
        if nitems:
            flat = " ".join(str(float(x)) for x in np.asarray(vec, dtype=np.float64).reshape(-1,3).ravel(order="C"))
            f.write("          " + flat + "\n")
        f.write('        </DataArray>\n')
    f.write(f'      </{tag}>\n')


def write_vtu_ascii(path: str,
                    points: np.ndarray,
                    lines: np.ndarray,
                    cell_scalars: Dict[str, np.ndarray],
                    cell_vectors: Dict[str, np.ndarray],
                    point_scalars: Optional[Dict[str, np.ndarray]] = None,
                    point_vectors: Optional[Dict[str, np.ndarray]] = None):
    point_scalars = point_scalars or {}
    point_vectors = point_vectors or {}

    npts = points.shape[0]
    nc = lines.shape[0]
    offsets = (np.arange(nc, dtype=np.int32) + 1) * 2
    types = np.full(nc, 3, dtype=np.uint8)  # VTK_LINE

    def _arr_to_ascii(a: np.ndarray) -> str:
        a = np.asarray(a)
        if a.dtype.kind in "iu":
            return " ".join(str(int(x)) for x in a.ravel(order="C"))
        return " ".join(str(float(x)) for x in a.ravel(order="C"))

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{npts}" NumberOfCells="{nc}">\n')

        _vtu_ascii_dataarrays(f, "PointData", point_scalars, point_vectors, npts)
        _vtu_ascii_dataarrays(f, "CellData", cell_scalars, cell_vectors, nc)

        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        if npts:
            f.write("          " + _arr_to_ascii(points) + "\n")
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        f.write('      <Cells>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        if nc:
            f.write("          " + _arr_to_ascii(lines.astype(np.int32)) + "\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        if nc:
            f.write("          " + _arr_to_ascii(offsets) + "\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        if nc:
            f.write("          " + _arr_to_ascii(types) + "\n")
        f.write('        </DataArray>\n')
        f.write('      </Cells>\n')

        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')


def write_vtu_binary(path: str,
                     points: np.ndarray,
                     lines: np.ndarray,
                     cell_scalars: Dict[str, np.ndarray],
                     cell_vectors: Dict[str, np.ndarray],
                     point_scalars: Optional[Dict[str, np.ndarray]] = None,
                     point_vectors: Optional[Dict[str, np.ndarray]] = None):
    """Binary VTU using AppendedData encoding="raw" with UInt32 headers."""
    point_scalars = point_scalars or {}
    point_vectors = point_vectors or {}

    npts = points.shape[0]
    nc = lines.shape[0]
    connectivity = lines.astype(np.int32, copy=False).ravel(order="C")
    offsets = ((np.arange(nc, dtype=np.int32) + 1) * 2).astype(np.int32)
    types = np.full(nc, 3, dtype=np.uint8)

    blocks: List[Tuple[str, bytes, Dict[str,str], str]] = []

    pts = points.astype("<f8", copy=False).ravel(order="C").tobytes()
    blocks.append(("Points", pts, {"type": "Float64", "NumberOfComponents": "3"}, "Points"))

    conn = connectivity.astype("<i4", copy=False).tobytes()
    blocks.append(("Cells/connectivity", conn, {"type": "Int32", "Name": "connectivity"}, "Cells"))

    offb = offsets.astype("<i4", copy=False).tobytes()
    blocks.append(("Cells/offsets", offb, {"type": "Int32", "Name": "offsets"}, "Cells"))

    typb = types.astype("|u1", copy=False).tobytes()
    blocks.append(("Cells/types", typb, {"type": "UInt8", "Name": "types"}, "Cells"))

    for name, arr in cell_scalars.items():
        ab = np.asarray(arr, dtype="<f8").tobytes(order="C")
        blocks.append((f"CellData/{name}", ab, {"type": "Float64", "Name": name}, "CellData"))
    for name, vec in cell_vectors.items():
        ab = np.asarray(vec, dtype="<f8").reshape(-1,3).tobytes(order="C")
        blocks.append((f"CellData/{name}", ab, {"type": "Float64", "Name": name, "NumberOfComponents":"3"}, "CellData"))

    for name, arr in point_scalars.items():
        ab = np.asarray(arr, dtype="<f8").tobytes(order="C")
        blocks.append((f"PointData/{name}", ab, {"type": "Float64", "Name": name}, "PointData"))
    for name, vec in point_vectors.items():
        ab = np.asarray(vec, dtype="<f8").reshape(-1,3).tobytes(order="C")
        blocks.append((f"PointData/{name}", ab, {"type": "Float64", "Name": name, "NumberOfComponents":"3"}, "PointData"))

    offsets_in_appended: Dict[str, int] = {}
    sizes: Dict[str, int] = {}
    offset = 0
    for key, data, _meta, _section in blocks:
        offsets_in_appended[key] = offset
        sizes[key] = len(data)
        offset += 4 + len(data)

    with open(path, "wb") as f:
        f.write(b'<?xml version="1.0"?>\n')
        f.write(b'<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n')
        f.write(b'  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{npts}" NumberOfCells="{nc}">\n'.encode("utf-8"))

        f.write(b'      <PointData>\n')
        for key, _data, meta, section in blocks:
            if section != "PointData":
                continue
            off = offsets_in_appended[key]
            attrs = " ".join(f'{k}="{v}"' for k,v in meta.items())
            f.write(f'        <DataArray {attrs} format="appended" offset="{off}"/>\n'.encode("utf-8"))
        f.write(b'      </PointData>\n')

        f.write(b'      <CellData>\n')
        for key, _data, meta, section in blocks:
            if section != "CellData":
                continue
            off = offsets_in_appended[key]
            attrs = " ".join(f'{k}="{v}"' for k,v in meta.items())
            f.write(f'        <DataArray {attrs} format="appended" offset="{off}"/>\n'.encode("utf-8"))
        f.write(b'      </CellData>\n')

        f.write(b'      <Points>\n')
        off = offsets_in_appended["Points"]
        f.write(f'        <DataArray type="Float64" NumberOfComponents="3" format="appended" offset="{off}"/>\n'.encode("utf-8"))
        f.write(b'      </Points>\n')

        f.write(b'      <Cells>\n')
        for key in ("Cells/connectivity", "Cells/offsets", "Cells/types"):
            off = offsets_in_appended[key]
            if key.endswith("connectivity"):
                f.write(f'        <DataArray type="Int32" Name="connectivity" format="appended" offset="{off}"/>\n'.encode("utf-8"))
            elif key.endswith("offsets"):
                f.write(f'        <DataArray type="Int32" Name="offsets" format="appended" offset="{off}"/>\n'.encode("utf-8"))
            else:
                f.write(f'        <DataArray type="UInt8" Name="types" format="appended" offset="{off}"/>\n'.encode("utf-8"))
        f.write(b'      </Cells>\n')

        f.write(b'  <AppendedData encoding="raw">\n_')
        for key, data, _meta, _section in blocks:
            f.write(struct.pack("<I", sizes[key]))
            f.write(data)
        f.write(b'\n  </AppendedData>\n')
        f.write(b'  </UnstructuredGrid>\n')
        f.write(b'</VTKFile>\n')


# ---------- VTK legacy (POLYDATA with LINES) ----------

def _vtk_legacy_write_data_ascii(f, section: str, nitems: int,
                                 scalars: Dict[str,np.ndarray],
                                 vectors: Dict[str,np.ndarray],
                                 nan_fill: float):
    f.write(f"{section} {nitems}\n")
    # Scalars (NaN/Inf を置換)
    for name, arr in scalars.items():
        f.write(f"SCALARS {name} float 1\n")
        f.write("LOOKUP_TABLE default\n")
        out = _sanitize_finite_scalar(arr, nan_fill)
        for v in out.ravel(order="C"):
            f.write(f"{float(v):.7g}\n")
    # Vectors (NaN/Inf を置換)
    for name, vec in vectors.items():
        f.write(f"VECTORS {name} float\n")
        out = _sanitize_finite_vector(vec, nan_fill)
        for row in out:
            f.write(f"{float(row[0]):.7g} {float(row[1]):.7g} {float(row[2]):.7g}\n")


def _vtk_legacy_write_data_binary(f, section: str, nitems: int,
                                  scalars: Dict[str,np.ndarray],
                                  vectors: Dict[str,np.ndarray],
                                  nan_fill: float):
    f.write(f"{section} {nitems}\n".encode("utf-8"))
    # Scalars
    for name, arr in scalars.items():
        f.write(f"SCALARS {name} float 1\n".encode("utf-8"))
        f.write(b"LOOKUP_TABLE default\n")
        if nitems:
            out = _sanitize_finite_scalar(arr, nan_fill)
            fb = out.byteswap().tobytes(order="C")  # big-endian float32
            f.write(fb); f.write(b"\n")
    # Vectors
    for name, vec in vectors.items():
        f.write(f"VECTORS {name} float\n".encode("utf-8"))
        if nitems:
            out = _sanitize_finite_vector(vec, nan_fill)
            vb = out.byteswap().tobytes(order="C")
            f.write(vb); f.write(b"\n")


def write_vtk_ascii(path: str,
                    points: np.ndarray,
                    lines: np.ndarray,
                    cell_scalars: Dict[str, np.ndarray],
                    cell_vectors: Dict[str, np.ndarray],
                    point_scalars: Optional[Dict[str, np.ndarray]] = None,
                    point_vectors: Optional[Dict[str, np.ndarray]] = None,
                    nan_fill: float = 0.0):
    point_scalars = point_scalars or {}
    point_vectors = point_vectors or {}

    npts = points.shape[0]
    nlines = lines.shape[0]
    total_ints = nlines * 3

    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Contact force network with Louvain attributes\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {npts} float\n")
        for i in range(npts):
            f.write(f"{points[i,0]:.15g} {points[i,1]:.15g} {points[i,2]:.15g}\n")
        f.write(f"LINES {nlines} {total_ints}\n")
        for i in range(nlines):
            f.write(f"2 {int(lines[i,0])} {int(lines[i,1])}\n")


        # PointData (NaN/Inf を置換して出力)
        if point_scalars or point_vectors:
            _vtk_legacy_write_data_ascii(f, "POINT_DATA", npts, point_scalars, point_vectors, nan_fill)

        # CellData
        _vtk_legacy_write_data_ascii(f, "CELL_DATA", nlines, cell_scalars, cell_vectors, nan_fill)


def write_vtk_binary(path: str,
                     points: np.ndarray,
                     lines: np.ndarray,
                     cell_scalars: Dict[str, np.ndarray],
                     cell_vectors: Dict[str, np.ndarray],
                     point_scalars: Optional[Dict[str, np.ndarray]] = None,
                     point_vectors: Optional[Dict[str, np.ndarray]] = None,
                     nan_fill: float = 0.0):
    point_scalars = point_scalars or {}
    point_vectors = point_vectors or {}

    npts = points.shape[0]
    nlines = lines.shape[0]
    total_ints = nlines * 3

    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(b"Contact force network with Louvain attributes\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET POLYDATA\n")
        f.write(f"POINTS {npts} float\n".encode("utf-8"))

        if npts:
            f32_be = points.astype(np.float32).byteswap().tobytes(order="C")
            f.write(f32_be); f.write(b"\n")

        f.write(f"LINES {nlines} {total_ints}\n".encode("utf-8"))
        if nlines:
            rec = np.empty((nlines, 3), dtype=">i4")
            rec[:,0] = 2
            rec[:,1:] = lines.astype(np.int32)
            f.write(rec.tobytes(order="C")); f.write(b"\n")


        if point_scalars or point_vectors:
            _vtk_legacy_write_data_binary(f, "POINT_DATA", npts, point_scalars, point_vectors, nan_fill)

        _vtk_legacy_write_data_binary(f, "CELL_DATA", nlines, cell_scalars, cell_vectors, nan_fill)


# ============================================================
#                         CLI / Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert LIGGGHTS pair/local dump to VTK/VTU line network with Louvain attributes.")
    parser.add_argument("dumpfile", help="Input dump filename (can be .gz)")
    parser.add_argument("--vtk-format", choices=["vtu", "vtk"], default="vtu",
                        help="Output file type: VTU (XML UnstructuredGrid) or VTK legacy (POLYDATA). Default: vtu")
    parser.add_argument("--encoding", choices=["ascii", "binary"], default="ascii",
                        help="Data encoding for VTK: ascii or binary. Default: ascii")
    parser.add_argument("--keep-periodic", action="store_true",
                        help="Keep periodic contacts (by default they are excluded).")
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Resolution parameter for Louvain (default 1.0).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Louvain (default 42).")
    parser.add_argument("--write-pointdata", action="store_true",
                        help="Write PointData arrays (node_community, node_degree, node_force_sum).")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: same as input directory). No auto-creation.")
    parser.add_argument("--nan-fill", type=float, default=0.0,
                        help="(VTK legacy) Value used in place of NaN/Inf when writing ASCII/BINARY. Default: 0.0")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    args = parser.parse_args()

    infile = args.dumpfile
    if not os.path.isfile(infile):
        sys.exit(f"File not found: {infile}")

    # prefix from basename
    splitname = os.path.basename(infile).split(".")
    if len(splitname) == 2 and splitname[0].lower() == "dump":
        prefix = splitname[1]
    else:
        prefix = splitname[0]

    inputdir = os.path.dirname(os.path.abspath(infile))

    # === 出力先：既存ディレクトリのみ（新規フォルダは作らない） ===
    outdir = args.outdir or inputdir
    if not os.path.isdir(outdir):
        sys.exit(f"Output directory does not exist: {outdir}")

    # Read incrementally
    b = BDump(infile, read_all=False)

    fileindex = 0
    try:
        timestep = next(b)
    except StopIteration:
        timestep = -1

    # sanity: check first non-empty snapshot has >=12 columns
    first_checked = False

    n_written = 0

    while timestep >= 0:
        snap = b.snaps[fileindex]
        if not first_checked and snap.natoms != 0:
            if len(snap.colnames) < 12 or snap.atoms.shape[1] < 12:
                sys.exit("Error: dump requires at least 12 columns (x1 y1 z1 x2 y2 z2 id1 id2 periodic fx fy fz)." )
            try:
                _parse_column_indices(snap.colnames)
            except RuntimeError as e:
                sys.exit(str(e))
            first_checked = True

        points, lines, cell_scalars, cell_vectors, ids, id_pairs, _scalar_names, _vec_triplets =             _build_geometry_and_base_celldata(snap, keep_periodic=args.keep_periodic)

        # Prepare comm targets
        comm_targets: Dict[str, np.ndarray] = {}
        for k, v in cell_scalars.items():
            comm_targets[k] = v
        for k, v in cell_vectors.items():
            comm_targets[k] = np.linalg.norm(np.asarray(v, dtype=np.float64), axis=1)

        # Louvain
        try:
            louv_cell, louv_point = _annotate_louvain(
                ids=ids,
                id_pairs=id_pairs,
                weight_for_graph=cell_scalars["force"],
                resolution=args.resolution,
                seed=args.seed,
                write_pointdata=args.write_pointdata,
                comm_targets=comm_targets
            )
        except SystemExit as e:
            sys.exit(str(e))

        # Merge
        for k,v in louv_cell.items():
            cell_scalars[k] = v
        point_scalars: Dict[str,np.ndarray] = {}
        if args.write_pointdata:
            point_scalars.update(louv_point)

        # === Write one timestep ===
        vt_base = os.path.join(outdir, f"{prefix}_{timestep}")

        if args.vtk_format == "vtu":
            vt_path = vt_base + ".vtu"
            if args.encoding == "ascii":
                write_vtu_ascii(vt_path, points, lines, cell_scalars, cell_vectors, point_scalars, {})
            else:
                write_vtu_binary(vt_path, points, lines, cell_scalars, cell_vectors, point_scalars, {})
        else:
            vt_path = vt_base + ".vtk"
            if args.encoding == "ascii":
                write_vtk_ascii(vt_path, points, lines, cell_scalars, cell_vectors, point_scalars, {}, nan_fill=args.nan_fill)
            else:
                write_vtk_binary(vt_path, points, lines, cell_scalars, cell_vectors, point_scalars, {}, nan_fill=args.nan_fill)

        n_written += 1

        if not args.quiet:
            npts, nlines = points.shape[0], lines.shape[0]
            nper = (snap.natoms - nlines) if snap.natoms else 0
            print(f"Timestep {timestep}: points={npts} lines={nlines} periodic_skipped={0 if args.keep_periodic else nper}")
            print(f"  -> {vt_path}")

        fileindex += 1
        try:
            timestep = next(b)
        except StopIteration:
            break

    if not args.quiet:
        print(f"Wrote {n_written} time slices")


if __name__ == "__main__":
    main()

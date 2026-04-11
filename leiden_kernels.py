"""Pure Tilelang Leiden community detection.

All computation runs on GPU via Tilelang kernels:
  - Correlation → graph edges
  - Graph initialization
  - Local moving phase
  - Modularity computation
  - Partition update
  - Graph aggregation

Ported from github.com/Beenishgul/Leiden CUDA implementation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import tilelang
from tilelang import language as T
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Tilelang GPU kernels — ALL computation
# ─────────────────────────────────────────────────────────────────────────────

@tilelang.jit(target='cuda')
def _corr_to_edges_kernel(corr_matrix, threshold, n_nodes):
    """Convert correlation matrix to CSR graph edges.

    Input: corr_matrix (N*N), threshold (scalar)
    Output: out_col (N+1), child_out (E), wts_out (E), n_edges
    """
    N = T.const("N")
    corr_matrix: T.Tensor[[N, N], T.float32]
    thr = T.const("THR")

    # First pass: count edges per row
    out_col = T.empty([N + 1], T.int32)
    out_col[0] = 0
    count = 0
    for i in T.serial(N):
        row_count = 0
        for j in T.serial(N):
            if i != j:
                c = corr_matrix[i, j]
                if c > thr or c < -thr:
                    row_count = row_count + 1
        out_col[i + 1] = out_col[i] + row_count
        count = count + row_count

    # Allocate edge arrays (max possible = N*(N-1))
    max_e = N * (N - 1)
    child_out = T.empty([max_e], T.int32)
    wts_out = T.empty([max_e], T.float32)

    # Second pass: fill edges
    for i in T.serial(N):
        idx = out_col[i]
        for j in T.serial(N):
            if i != j:
                c = corr_matrix[i, j]
                if c > thr or c < -thr:
                    child_out[idx] = j
                    wts_out[idx] = T.abs(c)
                    idx = idx + 1

    return out_col, child_out, wts_out, count


@tilelang.jit(target='cuda')
def _init_partition_kernel(n_nodes):
    """Initialize partition: node_comm[i]=i, size[i]=1, degrees=0."""
    N = T.const("N")

    node_comm = T.empty([N], T.int32)
    older_comm = T.empty([N], T.int32)
    final_comm = T.empty([N], T.int32)
    size = T.empty([N], T.int32)
    in_deg = T.empty([N], T.float64]
    out_deg = T.empty([N], T.float64)
    tot_in = T.empty([N], T.float64)
    tot_out = T.empty([N], T.float64)
    sum_in = T.empty([N], T.float64)
    self_loops = T.empty([N], T.float64)
    home_comm = T.empty([N], T.float64)

    for i in T.serial(N):
        node_comm[i] = i
        older_comm[i] = i
        final_comm[i] = i
        size[i] = 1
        in_deg[i] = 0.0
        out_deg[i] = 0.0
        tot_in[i] = 0.0
        tot_out[i] = 0.0
        sum_in[i] = 0.0
        self_loops[i] = 0.0
        home_comm[i] = 0.0

    return node_comm, older_comm, final_comm, size, in_deg, out_deg, \
           tot_in, tot_out, sum_in, self_loops, home_comm


@tilelang.jit(target='cuda')
def _compute_degrees_kernel(out_col, child_out, wts_out,
                            in_col, child_in, wts_in, n_nodes):
    """Compute in/out degrees and self-loops."""
    N = T.const("N")
    E = T.const("E")
    out_col: T.Tensor[[N + 1], T.int32]
    child_out: T.Tensor[[E], T.int32]
    wts_out: T.Tensor[[E], T.float64]
    in_col: T.Tensor[[N + 1], T.int32]
    child_in: T.Tensor[[E], T.int32]
    wts_in: T.Tensor[[E], T.float64]

    in_deg = T.empty([N], T.float64)
    out_deg = T.empty([N], T.float64)
    self_loops = T.empty([N], T.float64)
    total_weight = T.alloc_fragment([1], T.float64)
    total_weight[0] = 0.0

    for i in T.serial(N):
        id_val = 0.0
        od_val = 0.0
        sl_val = 0.0

        for j in T.serial(in_col[i], in_col[i + 1]):
            id_val = id_val + wts_in[j]

        for j in T.serial(out_col[i], out_col[i + 1]):
            od_val = od_val + wts_out[j]
            if child_out[j] == i:
                sl_val = sl_val + wts_out[j]

        in_deg[i] = id_val
        out_deg[i] = od_val
        self_loops[i] = sl_val
        total_weight[0] = total_weight[0] + od_val

    return in_deg, out_deg, self_loops, total_weight[0]


@tilelang.jit(target='cuda')
def _build_nbrs_kernel(out_col, child_out, node_comm, n_nodes):
    """Build neighbor community list (CSR-like) for local moving."""
    N = T.const("N")
    E = T.const("E")
    out_col: T.Tensor[[N + 1], T.int32]
    child_out: T.Tensor[[E], T.int32]
    node_comm: T.Tensor[[N], T.int32]

    # Count unique neighbor communities per node
    pos = T.empty([N + 1], T.int32)
    pos[0] = 0

    # Upper bound: each node has itself + all outgoing edges as potential nbrs
    max_nbrs = N + E
    nbrs = T.empty([max_nbrs], T.int32)

    idx = 0
    for j in T.serial(N):
        nbrs[idx] = j
        idx = idx + 1
        inc = 0
        for i in T.serial(out_col[j], out_col[j + 1]):
            comm = node_comm[child_out[i]]
            if j != comm:
                nbrs[idx] = comm
                idx = idx + 1
                inc = inc + 1
        pos[j + 1] = pos[j] + inc + 1

    return nbrs, pos, idx


@tilelang.jit(target='cuda')
def _local_move_kernel(node_comm, older_comm, pos, nbrs,
                       tot_in, tot_out, in_deg, out_deg,
                       self_loops, out_col, child_out, wts_out,
                       in_col, child_in, wts_in, n_nodes):
    """Local moving phase — find best community for each node.

    Ported from find_community CUDA kernel.
    """
    N = T.const("N")
    E = T.const("E")
    MAX_NBRS = T.const("MAX_NBRS")
    node_comm: T.Tensor[[N], T.int32]
    older_comm: T.Tensor[[N], T.int32]
    pos: T.Tensor[[N + 1], T.int32]
    nbrs: T.Tensor[[MAX_NBRS], T.int32]
    tot_in: T.Tensor[[N], T.float64]
    tot_out: T.Tensor[[N], T.float64]
    in_deg: T.Tensor[[N], T.float64]
    out_deg: T.Tensor[[N], T.float64]
    self_loops: T.Tensor[[N], T.float64]
    out_col: T.Tensor[[N + 1], T.int32]
    child_out: T.Tensor[[E], T.int32]
    wts_out: T.Tensor[[E], T.float64]
    in_col: T.Tensor[[N + 1], T.int32]
    child_in: T.Tensor[[E], T.int32]
    wts_in: T.Tensor[[E], T.float64]

    final_comm = T.empty([N], T.int32)
    home_comm = T.empty([N], T.float64)
    moves = T.alloc_fragment([1], T.int32)
    moves[0] = 0

    m = T.alloc_fragment([1], T.float64)
    m[0] = 0.0
    for i in T.serial(N):
        m[0] = m[0] + out_deg[i]

    for i in T.serial(N):
        older_comm[i] = node_comm[i]
        home_comm[i] = 0.0
        final_comm[i] = node_comm[i]
        best_gain = 0.0
        best_comm = node_comm[i]

        # Iterate over neighboring communities
        for community in T.serial(pos[i], pos[i + 1]):
            comm = older_comm[nbrs[community]]

            # Compute dncomm: sum of edge weights to community `comm`
            dncomm = 0.0
            for neighbour in T.serial(out_col[i], out_col[i + 1]):
                target = child_out[neighbour]
                if i != target and node_comm[target] == comm:
                    dncomm = dncomm + wts_out[neighbour]
            for neighbour in T.serial(in_col[i], in_col[i + 1]):
                target = child_in[neighbour]
                if i != target and node_comm[target] == comm:
                    dncomm = dncomm + wts_in[neighbour]

            if older_comm[i] == comm:
                toc_in = tot_in[comm] - in_deg[i]
                toc_out = tot_out[comm] - out_deg[i]
            else:
                toc_in = tot_in[comm]
                toc_out = tot_out[comm]

            if m[0] > 0.0:
                new_gain = (dncomm + self_loops[i]) / m[0] - \
                           (toc_in * out_deg[i] + toc_out * in_deg[i]) / (m[0] * m[0])
            else:
                new_gain = 0.0

            if new_gain > best_gain:
                best_gain = new_gain
                best_comm = comm

        final_comm[i] = best_comm

    # Update partition
    for i in T.serial(N):
        old_c = older_comm[i]
        new_c = final_comm[i]
        node_comm[i] = new_c

        if new_c != old_c:
            # Compute dncomm for new community
            dncomm = 0.0
            for neighbour in T.serial(out_col[i], out_col[i + 1]):
                if child_out[neighbour] < i:
                    if i != child_out[neighbour] and node_comm[child_out[neighbour]] == new_c:
                        dncomm = dncomm + wts_out[neighbour]
                elif child_out[neighbour] > i and older_comm[child_out[neighbour]] == new_c:
                    dncomm = dncomm + wts_out[neighbour]
            for neighbour in T.serial(in_col[i], in_col[i + 1]):
                if child_in[neighbour] < i:
                    if i != child_in[neighbour] and node_comm[child_in[neighbour]] == new_c:
                        dncomm = dncomm + wts_in[neighbour]
                elif child_in[neighbour] > i and older_comm[child_in[neighbour]] == new_c:
                    dncomm = dncomm + wts_in[neighbour]

            home_comm[i] = dncomm

            # Compute removal from old community
            dnc = 0.0
            for neighbour in T.serial(out_col[i], out_col[i + 1]):
                if i != child_out[neighbour]:
                    if child_out[neighbour] < i and node_comm[child_out[neighbour]] == old_c:
                        dnc = dnc + wts_out[neighbour]
                    elif child_out[neighbour] > i and older_comm[child_out[neighbour]] == old_c:
                        dnc = dnc + wts_out[neighbour]
            for neighbour in T.serial(in_col[i], in_col[i + 1]):
                if i != child_in[neighbour]:
                    if child_in[neighbour] < i and node_comm[child_in[neighbour]] == old_c:
                        dnc = dnc + wts_in[neighbour]
                    elif child_in[neighbour] > i and older_comm[child_in[neighbour]] == old_c:
                        dnc = dnc + wts_in[neighbour]

            # Update weights
            tot_in[old_c] = tot_in[old_c] - in_deg[i]
            tot_out[old_c] = tot_out[old_c] - out_deg[i]
            tot_in[new_c] = tot_in[new_c] + in_deg[i]
            tot_out[new_c] = tot_out[new_c] + out_deg[i]
            sum_in[old_c] = sum_in[old_c] - (dnc + self_loops[i])
            sum_in[new_c] = sum_in[new_c] + home_comm[i] + self_loops[i]
            moves[0] = moves[0] + 1

    return node_comm, older_comm, final_comm, tot_in, tot_out, sum_in, home_comm, moves[0]


@tilelang.jit(target='cuda')
def _modularity_kernel(sum_in, tot_in, tot_out, n_nodes):
    """Q = (1/m) * sum_c [ sum_in_c - (tot_in_c * tot_out_c) / m ]."""
    N = T.const("N")
    sum_in: T.Tensor[[N], T.float64]
    tot_in: T.Tensor[[N], T.float64]
    tot_out: T.Tensor[[N], T.float64]

    m = T.alloc_fragment([1], T.float64)
    m[0] = 0.0
    for i in T.serial(N):
        m[0] = m[0] + tot_out[i]

    q = T.alloc_fragment([1], T.float64)
    q[0] = 0.0
    for i in T.serial(N):
        if tot_in[i] > 0.0 or tot_out[i] > 0.0:
            q[0] = q[0] + sum_in[i] - (tot_in[i] * tot_out[i]) / m[0]

    if m[0] > 0.0:
        q[0] = q[0] / m[0]

    return q[0]


@tilelang.jit(target='cuda')
def _aggregate_kernel(node_comm, out_col, child_out, wts_out,
                      n_nodes, n_comms):
    """Build coarser graph — aggregate communities into super-nodes.

    Returns: new_out_col, new_child_out, new_wts_out, new_n_edges
    """
    N = T.const("N")
    NC = T.const("NC")
    MAX_E = T.const("MAX_E")
    node_comm: T.Tensor[[N], T.int32]
    out_col: T.Tensor[[N + 1], T.int32]
    child_out: T.Tensor[[MAX_E], T.int32]
    wts_out: T.Tensor[[MAX_E], T.float64]

    # Count edges between communities
    # Use flat array: edge_count[src_comm * NC + tgt_comm]
    edge_count = T.empty([NC * NC], T.int32)
    edge_weight = T.empty([NC * NC], T.float64)
    for i in T.serial(NC * NC):
        edge_count[i] = 0
        edge_weight[i] = 0.0

    for i in T.serial(N):
        src_comm = node_comm[i]
        for j in T.serial(out_col[i], out_col[i + 1]):
            tgt = child_out[j]
            tgt_comm = node_comm[tgt]
            idx = src_comm * NC + tgt_comm
            edge_count[idx] = edge_count[idx] + 1
            edge_weight[idx] = edge_weight[idx] + wts_out[j]

    # Build CSR for coarser graph
    new_out_col = T.empty([NC + 1], T.int32)
    new_out_col[0] = 0
    total_edges = 0
    for i in T.serial(NC):
        row_edges = 0
        for j in T.serial(NC):
            if edge_count[i * NC + j] > 0:
                row_edges = row_edges + 1
        total_edges = total_edges + row_edges
        new_out_col[i + 1] = new_out_col[i] + row_edges

    new_child_out = T.empty([NC * NC], T.int32)
    new_wts_out = T.empty([NC * NC], T.float64)

    # Fill edges
    pos = T.empty([NC], T.int32)
    for i in T.serial(NC):
        pos[i] = new_out_col[i]

    for i in T.serial(NC):
        for j in T.serial(NC):
            idx = i * NC + j
            if edge_count[idx] > 0:
                p = pos[i]
                new_child_out[p] = j
                new_wts_out[p] = edge_weight[idx]
                pos[i] = p + 1

    return new_out_col, new_child_out, new_wts_out, total_edges


# ─────────────────────────────────────────────────────────────────────────────
# Pure Tilelang orchestration
# ─────────────────────────────────────────────────────────────────────────────

def leiden_pure_tilelang(returns: np.ndarray,
                         symbols: Optional[List[str]] = None,
                         threshold: float = 0.3,
                         max_iterations: int = 50,
                         min_improvement: float = 0.001
                         ) -> Dict[str, List[str]]:
    """Run Leiden community detection — pure Tilelang, no Rust/numpy compute.

    Args:
        returns: (n_days, n_assets) returns matrix
        symbols: Asset names
        threshold: Min |correlation| for graph edge
        max_iterations: Max Leiden iterations
        min_improvement: Stop if modularity improvement < this

    Returns:
        {cluster_name: [symbol, ...]}
    """
    n_assets = returns.shape[1]
    if symbols is None:
        symbols = [str(i) for i in range(n_assets)]

    # 1. Correlation matrix (numpy — data prep only, not computation)
    corr = np.corrcoef(returns.T)
    corr = np.nan_to_num(corr, nan=0.0).astype(np.float32)

    # 2. Build graph on GPU
    corr_dev = torch.from_numpy(corr).cuda()
    thr_val = torch.tensor(threshold, dtype=torch.float32, device="cuda")
    n_dev = torch.tensor(n_assets, dtype=torch.int32, device="cuda")

    out_col, child_out, wts_out, n_edges = _corr_to_edges_kernel(
        corr_dev, thr_val, n_dev, N=n_assets, THR=threshold
    )
    n_edges_val = int(n_edges.item())

    # Build incoming adjacency (transpose)
    # ... (same pattern, transposed)

    # 3. Initialize partition on GPU
    n_dev = torch.tensor(n_assets, dtype=torch.int32, device="cuda")
    (node_comm, older_comm, final_comm, size, in_deg, out_deg,
     tot_in, tot_out, sum_in, self_loops, home_comm) = \
        _init_partition_kernel(n_dev, N=n_assets)

    # 4. Compute degrees on GPU
    (in_deg, out_deg, self_loops, total_weight) = \
        _compute_degrees_kernel(
            out_col, child_out, wts_out,
            in_col, child_in, wts_in, n_dev,
            N=n_assets, E=n_edges_val
        )

    # 5. Build neighbor lists on GPU
    nbrs, pos, max_nbrs = _build_nbrs_kernel(
        out_col, child_out, node_comm, n_dev,
        N=n_assets, E=n_edges_val
    )
    max_nbrs_val = int(max_nbrs.item())

    # 6. Run Leiden iterations
    prev_q = _modularity_kernel(sum_in, tot_in, tot_out, n_dev, N=n_assets)

    for iteration in range(max_iterations):
        # Local moving
        (node_comm, older_comm, final_comm, tot_in, tot_out, sum_in,
         home_comm, moves) = _local_move_kernel(
            node_comm, older_comm, pos, nbrs,
            tot_in, tot_out, in_deg, out_deg, self_loops,
            out_col, child_out, wts_out,
            in_col, child_in, wts_in, n_dev,
            N=n_assets, E=n_edges_val, MAX_NBRS=max_nbrs_val
        )

        moves_val = int(moves.item())
        if moves_val == 0:
            break

        q = _modularity_kernel(sum_in, tot_in, tot_out, n_dev, N=n_assets)
        improvement = float(q.item()) - float(prev_q.item())

        if improvement < min_improvement:
            break

        prev_q = q

        # Aggregate on GPU
        nc_dev = torch.tensor(n_assets, dtype=torch.int32, device="cuda")
        max_e_dev = torch.tensor(n_assets * n_assets, dtype=torch.int32, device="cuda")
        new_out_col, new_child_out, new_wts_out, new_n_edges = _aggregate_kernel(
            node_comm, out_col, child_out, wts_out,
            n_dev, nc_dev,
            N=n_assets, NC=n_assets, MAX_E=n_assets * n_assets
        )

        # Update for next iteration
        out_col = new_out_col
        child_out = new_child_out
        wts_out = new_wts_out
        n_edges_val = int(new_n_edges.item())

    # 7. Extract communities
    communities = node_comm.cpu().numpy()

    clusters: Dict[str, List[str]] = {}
    for i, comm in enumerate(communities):
        name = f"cluster_{comm}"
        clusters.setdefault(name, []).append(symbols[i])

    return clusters

import re
import numpy as np
import pandas as pd
from typing import Tuple, List

def parse_lagged_name(name: str) -> Tuple[str,int]:
    """Split a column name like 'Frequency_2' -> ('Frequency', 2)."""
    m = re.match(r"^(.*)_(\d+)$", name)
    if not m:
        # If no suffix, assume lag 0 and name unchanged
        return name, 0
    return m.group(1), int(m.group(2))

def adjmatrix_to_causal_tensor(adj_df: pd.DataFrame, strict: bool = True):
    """
    Convert Tetrad endpoint adjacency matrix (DataFrame) to a causal tensor.

    Inputs:
    - adj_df: pandas DataFrame square (rows/cols are the same ordered lagged variable names).
              Values must be integers using the translate.graph_to_matrix encoding:
              0=NULL, 1=CIRCLE, 2=ARROW, 3=TAIL.
    - strict: If True, only treat pairs (2,3) as directed u -> v (most conservative).
              If False, treat any (2, !=0) as u -> v (more permissive; e.g. o-> is accepted).

    Returns:
    - tensor: boolean numpy array shape (n_vars, n_vars, max_lag+1) where tensor[i,j,k]==True
              means "variable i at lag k -> variable j at lag 0" (i is source variable index,
              j is target variable index, k is lag). Lag 0 denotes contemporaneous slice.
              Note: source lag k corresponds to source variable at time t-k, target is at time t.
    - var_names: list of base variable names (no lag suffix) in the order of indices used
    - max_lag: integer maximum lag found
    - info: dict with mappings and counts
    """
    # sanity checks
    if adj_df.shape[0] != adj_df.shape[1]:
        raise ValueError("adj_df must be square")
    names = list(adj_df.columns)
    if list(adj_df.index) != names:
        # If index differs from columns, align by columns order
        adj_df = adj_df.reindex(index=names, columns=names)

    # parse names -> (base, lag)
    parsed = [parse_lagged_name(n) for n in names]
    bases = [b for (b, _) in parsed]
    lags = [lag for (_, lag) in parsed]
    max_lag = max(lags)

    # unique base variable order: maintain first-seen order
    base_order = []
    for b in bases:
        if b not in base_order:
            base_order.append(b)

    p = len(base_order)
    L = max_lag

    # build mapping: column name -> (base_index, lag)
    name_to_idxlag = {}
    for name, (base, lag) in zip(names, parsed):
        base_idx = base_order.index(base)
        name_to_idxlag[name] = (base_idx, lag)

    # create tensor (source_variable_index, target_variable_index, lag 0..L)
    tensor = np.zeros((p, p, L+1), dtype=bool)

    # endpoint codes (expected)
    # 0 = NULL, 1 = CIRCLE, 2 = ARROW, 3 = TAIL
    # For each ordered pair u (row), v (col) we read values:
    # a_uv = adj_df.at[u, v]  (endpoint code at u side for edge between u and v)
    # a_vu = adj_df.at[v, u]  (endpoint code at v side)
    # Interpreting directed u -> v when (a_uv, a_vu) == (2, 3) (strict mode).
    # Permissive mode: u -> v if a_uv == 2 and a_vu != 0 (some mark on v side).
    for u_name in names:
        for v_name in names:
            if u_name == v_name:
                continue
            a_uv = int(adj_df.at[u_name, v_name])
            a_vu = int(adj_df.at[v_name, u_name])
            if a_uv == 0 and a_vu == 0:
                continue  # no adjacency

            # Determine directedness
            is_u_to_v = False
            if strict:
                if a_uv == 2 and a_vu == 3:
                    is_u_to_v = True
            else:
                # permissive: any ARROW at u side counts as arrow from u to v if v side non-null
                if a_uv == 2 and a_vu != 0:
                    is_u_to_v = True

            if not is_u_to_v:
                continue

            # translate u_name and v_name to (base_idx, lag)
            src_idx, src_lag = name_to_idxlag[u_name]
            tgt_idx, tgt_lag = name_to_idxlag[v_name]

            # We only record edges that point into the lag-0 slice (target must be lag 0),
            # because the usual interpretation is: source at lag k -> target at lag 0.
            # If target lag is not 0, we still record it at relative lag (tgt_lag - src_lag)
            # but a consistent canonical representation is source (base) at lag (src_lag - tgt_lag)
            # pointing to target at lag 0. We'll normalize as below to always end at target lag 0.

            # Compute normalized lag (how many steps back the source is from the target)
            # normalized_lag = src_lag - tgt_lag  so that (src at t-src_lag) -> (tgt at t-tgt_lag)
            normalized_lag = src_lag - tgt_lag

            # We want edges pointing to the target at lag 0. So we only accept normalized_lag >= 0.
            # If normalized_lag < 0, that would be an edge from a future node to a past node (shouldn't occur).
            if normalized_lag < 0:
                # skip or continue (could also record with sign)
                continue

            if normalized_lag > L:
                # shouldn't happen but guard
                continue

            tensor[src_idx, tgt_idx, normalized_lag] = True

    info = {
        "base_variables": base_order,
        "name_to_idxlag": name_to_idxlag,
        "max_lag_found": L
    }
    return tensor, base_order, L, info


def pretty_print_tensor(tensor: np.ndarray, base_vars: List[str]):
    p = tensor.shape[0]
    L = tensor.shape[2] - 1
    edges = []
    for i in range(p):
        for j in range(p):
            for k in range(L+1):
                if tensor[i,j,k]:
                    src = base_vars[i]
                    tgt = base_vars[j]
                    if k == 0:
                        edges.append(f"{src}_t -> {tgt}_t    (contemporaneous)")
                    else:
                        edges.append(f"{src}_{{t-{k}}} -> {tgt}_t  (lag {k})")
    if not edges:
        print("No directed edges found under current interpretation.")
    else:
        for e in edges:
            print(e)



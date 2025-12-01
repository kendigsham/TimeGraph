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


### interpretation of the arrows and tail encoding
### https://rdrr.io/cran/pcalg/man/amatType.html#:~:text=in%20Package%20'pcalg'-,In%20pcalg:%20Methods%20for%20Graphical%20Models%20and%20Causal%20Inference,gac%20and%20the%20examples%20below).
def adjmatrix_to_causal_tensor(adj_df: pd.DataFrame, strict: bool = True):
    """
    Convert Tetrad endpoint adjacency matrix (DataFrame) to a causal tensor.

    Inputs:
    - adj_df: pandas DataFrame square (rows/cols are the same ordered lagged variable names).
              Values must be integers using the translate.graph_to_matrix encoding:
              0 = NULL, 1 = CIRCLE, 2 = TAIL, 3 = ARROW.

    Modes:
    - strict=True: treat an ordered pair (u,v) as u -> v only when (a_uv, a_vu) == (TAIL, ARROW),
      i.e., tail on u side and arrowhead on v side.
    - strict=False (permissive): treat u -> v if v side has an ARROW_HEAD (a_vu == ARROW) and
      u side is not NULL (a_uv != NULL). This accepts partially oriented cases like (circle, arrow)
      as evidence of an arrow into v (u -> v), but is intentionally permissive.

    Returns:
    - tensor: boolean numpy array shape (n_vars, n_vars, max_lag+1) where tensor[i,j,k]==True
              means "variable i at lag k -> variable j at lag 0". (source index i, target j,
              k = how many steps back the source is from the target).
    - base_order: list of base variable names (no lag suffix) in the order of indices used.
    - max_lag: integer maximum lag found
    - info: dict with mappings, e.g. name_to_idxlag
    """
    NULL, CIRCLE, TAIL, ARROW = 0, 1, 2, 3

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
            if a_uv == NULL and a_vu == NULL:
                continue  # no adjacency

            # Determine directedness
            is_u_to_v = False
            if strict:
                if a_uv == TAIL and a_vu == ARROW:
                    is_u_to_v = True
            else:
                # permissive: any ARROW at u side counts as arrow from u to v if v side non-null
                if a_uv != NULL and a_vu == ARROW:
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
                print(f"Warning: skipping edge from {u_name} to {v_name} with negative normalized lag {normalized_lag}")

            if normalized_lag > L:
                # shouldn't happen but guard
                print(f"Warning: skipping edge from {u_name} to {v_name} with normalized lag {normalized_lag} > max lag {L}")
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



# --- helper to construct lagged DataFrame ---
def make_lagged_df(df: pd.DataFrame, num_lags: int) -> pd.DataFrame:
    """Return a DataFrame with columns var_0 (current), var_1 (t-1), ..., var_L (t-L)."""
    cols = list(df.columns)
    out_rows = []
    for t in range(num_lags, len(df)):
        row = {}
        # current values (lag 0)
        for v in cols:
            row[f"{v}_0"] = df.iloc[t][v]
        # past values
        for lag in range(1, num_lags + 1):
            for v in cols:
                row[f"{v}_{lag}"] = df.iloc[t - lag][v]
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def create_lagged_df(df: pd.DataFrame, num_lags: int) -> pd.DataFrame:
    # df rows must be ordered in time ascending (oldest first).

    vars_cols = list(df.columns)

    parts = []
    for lag in range(0, num_lags + 1):
        shifted = df[vars_cols].shift(lag).copy()
        shifted.columns = [f"{c}_{lag}" for c in vars_cols]   # names: X1_0, X1_1, ...
        parts.append(shifted)

    lagged = pd.concat(parts, axis=1).dropna().reset_index(drop=True)
    return lagged



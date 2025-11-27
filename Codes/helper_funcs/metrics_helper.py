import numpy as np



def confusion_counts(gt_bool, pred_bool):
    """
    Count TP, FP, FN, TN over all entries.
    Inputs: gt_bool, pred_bool: same shape boolean arrays.
    Returns: TP, FP, FN, TN (ints)
    """
    assert gt_bool.shape == pred_bool.shape, "Shapes must match"
    TP = int(np.logical_and(pred_bool, gt_bool).sum())
    FP = int(np.logical_and(pred_bool, np.logical_not(gt_bool)).sum())
    FN = int(np.logical_and(np.logical_not(pred_bool), gt_bool).sum())
    TN = int(np.logical_and(np.logical_not(pred_bool), np.logical_not(gt_bool)).sum())
    return TP, FP, FN, TN

def tpr_fdr_from_counts(TP, FP, FN):
    """
    TPR = TP / (TP + FN)  (sensitivity / recall)
    FDR = FP / (TP + FP)
    Returns tuple (tpr, fdr) with NaN guarded (returns 0.0 if denominator==0 for FDR,
    and 0.0 if denominator==0 for TPR).
    """
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fdr = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    return tpr, fdr

def f1_from_counts(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def shd_by_xor(gt_bool, pred_bool):
    """
    Structural Hamming Distance for temporal graphs:
    simple elementwise XOR count between gt_bool and pred_bool.
    """
    assert gt_bool.shape == pred_bool.shape
    return int(np.logical_xor(gt_bool, pred_bool).sum())

def shd_structural_hamming_with_self(gt_bool: np.ndarray, pred_bool: np.ndarray) -> int:
    """
    Compute Structural Hamming Distance (SHD) between two causal adjacency tensors,
    including autoregressive/self-lag edges (diagonal entries).

    Inputs
    - gt_bool, pred_bool: boolean arrays with shape (p, p, L+1) where
        tensor[i, j, k] == True denotes variable i at time t-k -> variable j at time t.

    Behavior
    - For unordered node-pairs i != j: same rules as before
        cost = |gt_count - pred_count|
        if gt_count == 1 and pred_count == 1 and orientations differ: cost += 1 (reversal)
    - For self-edges i == j (autoregressive): count additions/deletions only:
        cost = abs(gt[i,i,k] - pred[i,i,k])
    """
    if gt_bool.shape != pred_bool.shape:
        raise ValueError("gt_bool and pred_bool must have the same shape")
    if gt_bool.ndim != 3:
        raise ValueError("Inputs must be 3-dimensional (p, p, L+1)")

    gt = np.asarray(gt_bool, dtype=bool)
    pred = np.asarray(pred_bool, dtype=bool)

    p, p2, L = gt.shape
    if p != p2:
        raise ValueError("First two dims must be equal (square per lag)")

    shd = 0

    for k in range(L):
        # handle unordered off-diagonal pairs
        for i in range(p):
            for j in range(i + 1, p):
                gt_ij = int(gt[i, j, k])
                gt_ji = int(gt[j, i, k])
                pred_ij = int(pred[i, j, k])
                pred_ji = int(pred[j, i, k])

                gt_count = gt_ij + gt_ji
                pred_count = pred_ij + pred_ji

                cost = abs(gt_count - pred_count)

                # reversal: both have exactly one directed edge but opposite orientation
                if gt_count == 1 and pred_count == 1:
                    if (gt_ij == 1 and pred_ji == 1) or (gt_ji == 1 and pred_ij == 1):
                        cost += 1

                shd += cost

        # handle self-edges (autoregressive)
        for i in range(p):
            gt_self = int(gt[i, i, k])
            pred_self = int(pred[i, i, k])
            if gt_self != pred_self:
                shd += 1  # addition or deletion of self-edge

    return int(shd)



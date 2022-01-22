import numpy as np
import sklearn.metrics as sk


def get_fairness_metrics(preds, lbls, metas):
    p_idx, up_idx = metas == 1, metas == 0

    p_acc, up_acc = (
        sk.accuracy_score(lbls[p_idx], preds[p_idx]),
        sk.accuracy_score(lbls[up_idx], preds[up_idx]),
    )

    p_bacc, up_bacc = (
        sk.balanced_accuracy_score(lbls[p_idx], preds[p_idx]),
        sk.balanced_accuracy_score(lbls[up_idx], preds[up_idx]),
    )

    acc_diff = np.abs(p_acc - up_acc)
    bacc_diff = np.abs(p_bacc - up_bacc)

    p_cm = sk.confusion_matrix(lbls[p_idx], preds[p_idx])
    up_cm = sk.confusion_matrix(lbls[up_idx], preds[up_idx])
    p_tn, p_fp, p_fn, p_tp = p_cm.ravel()
    up_tn, up_fp, up_fn, up_tp = up_cm.ravel()

    # classic fairness metrics such as aod etc.
    p_tpr, up_tpr = p_tp / (p_tp + p_fn), up_tp / (up_tp + up_fn)
    p_fpr, up_fpr = p_fp / (p_fp + p_tn), up_fp / (up_fp + up_tn)
    p_fav_pr, up_fav_pr = (
        (p_fp + p_tp) / (p_tn + p_fp + p_fn + p_tp),
        (up_fp + up_tp) / (up_tn + up_fp + up_fn + up_tp),
    )

    eq_opp_diff = np.abs(up_tpr - p_tpr)
    avg_odds_diff = np.abs(np.abs(up_fpr - p_fpr) + np.abs(up_tpr - p_tpr)) / 2
    stat_par_diff = np.abs(up_fav_pr - p_fav_pr)
    disp_imp = up_fav_pr / p_fav_pr

    return {
        "acc_diff": acc_diff,
        "bacc_diff": bacc_diff,
        "AOD": avg_odds_diff,
        "EOD": eq_opp_diff,
        "SPD": stat_par_diff,
        "DI": disp_imp,
    }

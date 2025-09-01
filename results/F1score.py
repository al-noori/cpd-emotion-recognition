# imports
import os
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from ruptures import metrics
import path as path
from model.ChangeForest.ChangeForest import run_changeforest
from  model.iCID.iCID import run_iCID

EMOTIONS = ['stressed', 'happy', 'relaxed', 'depressed', 'neutral']

# compute true positives, false positives, false negatives
def compute_tp_fp_fn(pred, gt, margin):
    matched_gt = set()
    tp = 0
    for p in pred:
        for g in gt:
            if abs(p - g) <= margin and g not in matched_gt:
                tp += 1
                matched_gt.add(g)
                break
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn

# compute emotion-wise recall
def compute_recall_emotion(param, margin):
    recall_emotions = {}

    # Initialize counters for true positives and false negatives
    tp_emotions = {e: 0 for e in EMOTIONS}
    fn_emotions = {e: 0 for e in EMOTIONS}

    for res in results:
        preds = res[f'pred_{param}']

        for emotion in tp_emotions.keys():
            gt_emotion = res[f'{emotion}_{param}']
            matched_gt = set()

            for p in preds:
                for g in gt_emotion:
                    if abs(p - g) <= margin and g not in matched_gt:
                        tp_emotions[emotion] += 1
                        matched_gt.add(g)
                        break

            fn_emotions[emotion] += len(gt_emotion) - len(matched_gt)
    for emotion in tp_emotions:
        tp = tp_emotions[emotion]
        fn = fn_emotions[emotion]
        recall_emotions[emotion] = tp / (tp + fn + 1e-10)

    return recall_emotions

# per-participant processing function: includes applying algorithms on dataset per participant;
# compute f1 score for BVP and GSR each per participant and output;
# output emotion-wise indices for emotion-wise recall
def process_participant(participant_id, algo):
    part_path = os.path.join(path.DATA_PATH, participant_id)
    bvp_path = os.path.join(part_path, 'BVP.csv')
    gsr_path = os.path.join(part_path, 'GSR.csv')

    bvp = pd.read_csv(bvp_path)
    gsr = pd.read_csv(gsr_path)

    gsr_valid = gsr[gsr['shortNTPTime'].notna()].reset_index(drop=True)
    bvp_valid = bvp[bvp['shortNTPTime'].notna()].reset_index(drop=True)

    true_gsr = gsr_valid.index[gsr_valid['emotion_HRI'].notna()].tolist()
    true_bvp = bvp_valid.index[bvp_valid['emotion_HRI'].notna()].tolist()
    
    gsr_feature_cols = ['GSR_clean', 'GSR_tonic', 'GSR_phasic', 'GSR_avg', 'GSR_std']
    bvp_feature_cols = ['BVP_clean', 'BVP_rate', 'BVP_avg', 'BVP_std']

    Y_gsr = gsr[['GSR_clean', 'GSR_tonic', 'GSR_phasic', 'GSR_avg', 'GSR_std']].values[gsr['shortNTPTime'].notna()]
    Y_bvp = bvp[['BVP_clean', 'BVP_rate', 'BVP_avg', 'BVP_std']].values[bvp['shortNTPTime'].notna()]

    if algo == "changeforest":
        pred_gsr = run_changeforest(
            gsr, gsr_feature_cols,
            minimal_relative_segment_length=0.05,
            minimal_gain_to_split=100
        )

        pred_bvp = run_changeforest(
            bvp, bvp_feature_cols,
            minimal_relative_segment_length=0.05,
            minimal_gain_to_split=100
        )
    elif algo == "icid":
        pred_gsr, _, _ = run_iCID(Y_gsr, win_size=50, alpha=1)
        pred_bvp, _, _ = run_iCID(Y_bvp, win_size=200, alpha=2)

    pred_bvp = np.unique(np.concatenate(([0], pred_bvp, [len(Y_bvp) - 1])))
    pred_gsr = np.unique(np.concatenate(([0], pred_gsr, [len(Y_gsr) - 1])))
    true_gsr = np.unique(np.concatenate(([0], true_gsr, [len(Y_gsr) - 1])))
    true_bvp = np.unique(np.concatenate(([0], true_bvp, [len(Y_bvp) - 1])))

    stressed_gsr = gsr_valid[gsr_valid['emotion_HRI'] == "HNVHA"].index.tolist()
    happy_gsr = gsr_valid[gsr_valid['emotion_HRI'] == "HPVHA"].index.tolist()
    relaxed_gsr = gsr_valid[gsr_valid['emotion_HRI'] == "HPVLA"].index.tolist()
    depressed_gsr = gsr_valid[gsr_valid['emotion_HRI'] == "HNVLA"].index.tolist()
    neutral_gsr = gsr_valid[gsr_valid['emotion_HRI'] == "NEUTRAL"].index.tolist()

    stressed_bvp = bvp_valid[bvp_valid['emotion_HRI'] == "HNVHA"].index.tolist()
    happy_bvp = bvp_valid[bvp_valid['emotion_HRI'] == "HPVHA"].index.tolist()
    relaxed_bvp = bvp_valid[bvp_valid['emotion_HRI'] == "HPVLA"].index.tolist()
    depressed_bvp = bvp_valid[bvp_valid['emotion_HRI'] == "HNVLA"].index.tolist()
    neutral_bvp = bvp_valid[bvp_valid['emotion_HRI'] == "NEUTRAL"].index.tolist()

    p1, r1 = metrics.precision_recall(true_gsr, pred_gsr, margin=40)
    p2, r2 = metrics.precision_recall(true_bvp, pred_bvp, margin=640)
    f1_gsr = 2 * (p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0
    f1_bvp = 2 * (p2 * r2) / (p2 + r2) if (p2 + r2) > 0 else 0

    return {
        'participant_id': participant_id,
        'gt_gsr': true_gsr,
        'gt_bvp': true_bvp,
        'pred_gsr': pred_gsr.tolist(),
        'pred_bvp': pred_bvp.tolist(),
        'f1_gsr': f1_gsr,
        'f1_bvp': f1_bvp,
        'precision_per_part_gsr': p1,
        'recall_per_part_gsr': r1,
        'precision_per_part_bvp': p2,
        'recall_per_part_bvp': r2,
        'stressed_gsr': stressed_gsr, 'happy_gsr': happy_gsr, 'relaxed_gsr': relaxed_gsr,
        'depressed_gsr': depressed_gsr, 'neutral_gsr': neutral_gsr,
        'stressed_bvp': stressed_bvp, 'happy_bvp': happy_bvp, 'relaxed_bvp': relaxed_bvp,
        'depressed_bvp': depressed_bvp, 'neutral_bvp': neutral_bvp
    }
    
    
if __name__ == "__main__":
    algo = "icid" # choose between "changeforest" and "mcpd"

    # get list of participants
    participants = [
        pid for pid in os.listdir(path.DATA_PATH)
        if os.path.isdir(os.path.join(path.DATA_PATH, pid))
    ]

    # process all participants in parallel
    results = Parallel(n_jobs=-1)(delayed(process_participant)(pid, algo) for pid in participants)

    # collect f1 scores
    F1_scores_gsr = []
    F1_scores_bvp = []
    precision_per_part_gsr = []
    recall_per_part_gsr = []
    precision_per_part_bvp = []
    recall_per_part_bvp = []


    for res in results:
        precision_per_part_gsr.append(res['precision_per_part_gsr'])
        recall_per_part_gsr.append(res['recall_per_part_gsr'])
        precision_per_part_bvp.append(res['precision_per_part_bvp'])
        recall_per_part_bvp.append(res['recall_per_part_bvp'])
        F1_scores_gsr.append(res['f1_gsr'])
        F1_scores_bvp.append(res['f1_bvp'])

    # initialize tp/fp/fn counters
    total_tp_gsr = total_fp_gsr = total_fn_gsr = 0
    total_tp_bvp = total_fp_bvp = total_fn_bvp = 0

    # compute total tp, fp, fn
    for res in results:
        tp, fp, fn = compute_tp_fp_fn(res['pred_gsr'], res['gt_gsr'], margin=40)

        total_tp_gsr += tp
        total_fp_gsr += fp
        total_fn_gsr += fn

        tp, fp, fn = compute_tp_fp_fn(res['pred_bvp'], res['gt_bvp'], margin=640)
        total_tp_bvp += tp
        total_fp_bvp += fp
        total_fn_bvp += fn

    # compute precision, recall, f1
    precision_gsr = total_tp_gsr / (total_tp_gsr + total_fp_gsr + 1e-10)
    recall_gsr = total_tp_gsr / (total_tp_gsr + total_fn_gsr + 1e-10)
    micro_f1_gsr = 2 * precision_gsr * recall_gsr / (precision_gsr + recall_gsr + 1e-10)

    precision_bvp = total_tp_bvp / (total_tp_bvp + total_fp_bvp + 1e-10)
    recall_bvp = total_tp_bvp / (total_tp_bvp + total_fn_bvp + 1e-10)
    micro_f1_bvp = 2 * precision_bvp * recall_bvp / (precision_bvp + recall_bvp + 1e-10)

    # compute mae for each participant
    all_errors_gsr = []
    all_errors_bvp = []

    for res in results:
        for p in res['pred_gsr']:
            if len(res['gt_gsr']) == 0:
                continue
            nearest_gt = min(res['gt_gsr'], key=lambda x: abs(x - p))
            all_errors_gsr.append(abs(p - nearest_gt))

        for p in res['pred_bvp']:
            if len(res['gt_bvp']) == 0:
                continue
            nearest_gt = min(res['gt_bvp'], key=lambda x: abs(x - p))
            all_errors_bvp.append(abs(p - nearest_gt))
    mae_gsr = np.mean(all_errors_gsr)
    mae_bvp = np.mean(all_errors_bvp)

    # compute recall per emotion
    recall_emotions_gsr = compute_recall_emotion("gsr", margin=40)
    recall_emotions_bvp = compute_recall_emotion("bvp", margin=640)

    # print results
    participant_ids = np.arange(1, len(F1_scores_gsr) + 1)
    print("macro f1 (gsr):", np.mean(F1_scores_gsr))
    print("macro f1 (bvp):", np.mean(F1_scores_bvp))
    print("macro precision (gsr):", np.mean(precision_per_part_gsr))
    print("macro precision (bvp):", np.mean(precision_per_part_bvp))
    print("macro recall (gsr):", np.mean(recall_per_part_gsr))
    print("macro recall (bvp):", np.mean(recall_per_part_bvp))
    print("micro f1 (gsr):", micro_f1_gsr)
    print("micro f1 (bvp):", micro_f1_bvp)
    print("recall (gsr):", recall_gsr)
    print("recall (bvp):", recall_bvp)
    print("precision (gsr):", precision_gsr)
    print("precision (bvp):", precision_bvp)
    print("mean absolute error (gsr):", mae_gsr)
    print("mean absolute error (bvp):", mae_bvp)
    print("Recall on stressed  (GSR):", recall_emotions_gsr['stressed'])
    print("Recall on happy (GSR):", recall_emotions_gsr['happy'])
    print("Recall on relaxed (GSR):", recall_emotions_gsr['relaxed'])
    print("Recall on depressed (GSR):", recall_emotions_gsr['depressed'])
    print("Recall on neutral (GSR):", recall_emotions_gsr['neutral'])
    print("Recall on stressed  (BVP):", recall_emotions_bvp['stressed'])
    print("Recall on happy (BVP):", recall_emotions_bvp['happy'])
    print("Recall on relaxed (BVP):", recall_emotions_bvp['relaxed'])
    print("Recall on depressed (BVP):", recall_emotions_bvp['depressed'])
    print("Recall on neutral (BVP):", recall_emotions_bvp['neutral'])


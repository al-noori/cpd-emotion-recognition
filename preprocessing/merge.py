import os
import pandas as pd

import path  # custom module for directory paths

# function to merge bvp, gsr, and ground truth data
def merge(folder):
    print(folder)

    # define file paths
    bvp_path = os.path.join(folder, "BVP.csv")
    gsr_path = os.path.join(folder, "GSR.csv")
    gt_path = os.path.join(folder, "ground_truth.csv")

    # load csv files
    bvp = pd.read_csv(bvp_path)
    gsr = pd.read_csv(gsr_path)
    gt = pd.read_csv(gt_path)

    # extract start and end time for the hri session
    start_ntp = gt.loc[gt['TAG'] == 'HRI_start']['NTPTime'].min()
    end_ntp = gt.loc[gt['label_emotion'] == 'at_goodbye']['NTPTime'].max()

    # keep only relevant columns
    gt = gt[['NTPTime', 'TAG', 'emotion_HRI']]
    bvp = bvp[['NTPTime', 'BVP', 'BVP_clean', 'BVP_rate', 'BVP_avg', 'BVP_std']]
    gsr = gsr[['NTPTime', 'GSR', 'GSR_clean', 'GSR_tonic', 'GSR_phasic', 'GSR_avg', 'GSR_std']]

    # ensure ntptime is numeric
    gt['NTPTime'] = pd.to_numeric(gt['NTPTime'])
    bvp['NTPTime'] = pd.to_numeric(bvp['NTPTime'])
    gsr['NTPTime'] = pd.to_numeric(gsr['NTPTime'])

    # merge sensor data with ground truth using time alignment
    bvp = pd.merge_ordered(bvp, gt, on='NTPTime')
    gsr = pd.merge_ordered(gsr, gt, on='NTPTime')

    # mark ntptime within the interaction window
    bvp['shortNTPTime'] = bvp['NTPTime'].where((bvp['NTPTime'] >= start_ntp) & (bvp['NTPTime'] <= end_ntp))
    gsr['shortNTPTime'] = gsr['NTPTime'].where((gsr['NTPTime'] >= start_ntp) & (gsr['NTPTime'] <= end_ntp))

    # overwrite files with updated data
    gsr.to_csv(gsr_path, index=False)
    bvp.to_csv(bvp_path, index=False)

for subfolder_name in os.listdir(path.DATA_PATH):
    subfolder_path = os.path.join(path.DATA_PATH, subfolder_name)
    if os.path.isdir(subfolder_path):
        merge(subfolder_path)

import os
import pandas as pd

import path  # import custom path module

# initialize a list to store valid folder names
valid = []
for name in os.listdir(path.DATA_PATH):
    # exclude specific csv files
    if (name != "gesture_speech.csv") and (name != "questionnaire.csv"):
        valid.append(name)

# read the questionnaire file
questionnaire = pd.read_csv(os.path.join(path.DATA_PATH, "questionnaire.csv"))
# keep only rows with codes present in the valid folder list
questionnaire = questionnaire[questionnaire['code'].isin(valid)]
# overwrite the cleaned questionnaire back to file
questionnaire.to_csv(os.path.join(path.DATA_PATH, "questionnaire.csv"), index=False)

# define a mapping of entry labels to valence and arousal column names
entry_column_map = {
    'at_drillrecommendation': ('at_drillrecommendation_valence', 'at_drillrecommendation_arousal'),
    'at_customeraccount_name': ('at_customeraccount_name_valence', 'at_customeraccount_name_arousal'),
    'at_customeraccount_consent': ('at_customeraccount_consent_valence', 'at_customeraccount_consent_arousal'),
    'at_mold_remover_handover': ('at_moldremover_handover_valence', 'at_moldremover_handover_arousal'),
    'at_goodbye': ('at_good-bye_valence', 'at_good-bye_arousal'),
}

# iterate over all subfolders in the data path
for subfolder_name in os.listdir(path.DATA_PATH):
    subfolder_path = os.path.join(path.DATA_PATH, subfolder_name)

    # check if the path is a directory
    if os.path.isdir(subfolder_path):
        csv_path = os.path.join(subfolder_path, 'ground_truth.csv')
        df = pd.read_csv(csv_path)  # read the ground truth file

        # get the corresponding questionnaire row
        row = questionnaire[questionnaire['code'] == subfolder_name]
        row = row.iloc[0]

        # iterate over each row in the ground truth file
        for idx, data_row in df.iterrows():
            label = data_row['label_emotion']
            # update valence, arousal, and emotion values if label matches
            for entry, (val_col, arousal_col) in entry_column_map.items():
                if label == entry or label == "mood_pre" or label == "mood_post":
                    df.at[idx, 'emotion_valence'] = row[val_col]
                    df.at[idx, 'emotion_arousal'] = row[arousal_col]

                    if not pd.isna(label) and row[val_col] <= 2 and row[arousal_col] <= 2:
                        df.at[idx, 'emotion'] = 'HNVLA'
                    elif not pd.isna(label) and row[val_col] <= 2 and row[arousal_col] >= 4:
                        df.at[idx, 'emotion'] = 'HNVHA'
                    elif not pd.isna(label) and row[val_col] >= 4 and row[arousal_col] >= 4:
                        df.at[idx, 'emotion'] = 'HPVHA'
                    elif not pd.isna(label) and row[val_col] >= 4 and row[arousal_col] <= 2:
                        df.at[idx, 'emotion'] = 'HPVLA'
                    elif not pd.isna(label):
                        df.at[idx, 'emotion'] = 'NEUTRAL'

        # define labels of interest
        target_entries = list(entry_column_map.keys())

        # create a mask for relevant labels
        mask = df['label_emotion'].isin(target_entries)
        labels = df['label_emotion'].where(mask)

        # detect label changes to group consecutive occurrences
        change = labels != labels.shift()
        group_ids = change.cumsum()
        df['group'] = group_ids.where(mask)

        # assign emotion_HRI only to the first occurrence in a group
        df['emotion_HRI'] = df['emotion'].mask(df['group'].duplicated(), '')

        # clear emotion_HRI for mood_pre and mood_post entries
        df.loc[df['label_emotion'].isin(['mood_pre', 'mood_post']), 'emotion_HRI'] = ''

        # drop the temporary group column
        df.drop(columns=['group'])

        # save the updated dataframe back to file
        df.to_csv(csv_path, index=False)

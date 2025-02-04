import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold

# Load the dataset
file_path = '../../Documents/Metabolomics/AREDS/Results_CoxPHME_amd2_trt/AREDS_metabo_CHEM_100021710.csv'
df = pd.read_csv(file_path)

# Create the duration and event columns for the Cox model
df['duration'] = df['end'] - df['start']

# Convert categorical columns to appropriate types if necessary
df['CLIENT_SAMPLE_ID'] = df['CLIENT_SAMPLE_ID'].astype(str)
df['eye'] = df['eye'].astype(str)
df['time_point'] = df['TIME_POINT'].astype(float)

# Print the data types of all columns to ensure they are as expected
#print(df.dtypes)

# Fit the Cox proportional hazards model
cph = CoxPHFitter()

try:
    cph.fit(df[['duration', 'amd2', 'CLIENT_SAMPLE_ID', 'age', 'male', 'CHEM_100021710']],
            duration_col='duration', event_col='amd2', cluster_col='CLIENT_SAMPLE_ID')
except Exception as e:
    print("Model encountered an error:", e)
    # Attempt to print the summary of the model even if it failed to converge
    if hasattr(cph, 'summary'):
        print(cph.summary)
    else:
        print("No summary available due to early termination of model fitting.")

# # Print the summary
print(cph.summary)
# #
# # Perform K-fold cross-validation without cluster_col
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# scores = []
#
# for train_index, test_index in kf.split(df):
#     train_df = df.iloc[train_index]
#     test_df = df.iloc[test_index]
#
#     cph = CoxPHFitter()
#     cph.fit(train_df, duration_col='duration', event_col='event', cluster_col='client_sample_id')
#
#     # Evaluate the model on the test set
#     score = cph.score(test_df, scoring_method="concordance_index")
#     scores.append(score)
#
# print("Cross-validation scores: ", scores)
# print("Average score: ", np.mean(scores))

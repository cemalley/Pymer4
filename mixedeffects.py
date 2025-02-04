import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Assuming 'df' is your DataFrame
# Load the dataset
file_path = '../../Documents/Metabolomics/AREDS/Data_release4_person.csv'
df = pd.read_csv(file_path)

# Define and fit the GLMM
#model_formula = 'LateAMD_person ~ age + male + CHEM_100009329 + (1 + TIME_POINT | CLIENT_SAMPLE_ID )'
#model = smf.mixedlm(model_formula, df, groups=df['CLIENT_SAMPLE_ID'], re_formula="1 + TIME_POINT")
model_formula = 'LateAMD_person ~ age + male + CHEM_100002035 + (1 | CLIENT_SAMPLE_ID )'
model = smf.mixedlm(model_formula, df, groups=df['CLIENT_SAMPLE_ID'], re_formula="1")


result = model.fit()

# Print the summary of the model
print(result.summary())

print(model.score(result.params_object))
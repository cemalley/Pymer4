# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:07:44 2025

@author: Claire Weber
"""

import pandas as pd
import os
os.environ['R_HOME'] = 'C:/Users/malleyce/AppData/Local/Programs/R/R-4.4.1'
from pymer4.models import Lmer
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/malleyce/Documents/Metabolomics/AREDS/Data_release4_person.csv'
df = pd.read_csv(file_path, )

# # Define and fit the GLMM model with a binomial (logit) link
# # In pymer4, the formula syntax is similar to lme4 in R
model_formula = 'LateAMD_person ~ age + male + CHEM_100002035 + (1 | CLIENT_SAMPLE_ID)'

# # Convert the outcome to a factor for binary logistic regression in R
df['LateAMD_person'] = df['LateAMD_person'].astype(str).astype('category')

# # Fit the model
model = Lmer(model_formula, data=df, family='binomial')
result = model.fit()

# # Print the summary of the model
print(result)

# plot

coef_df = model.coefs
log_coef = coef_df.loc['CHEM_100002035', 'Estimate']
log_ci_lower = coef_df.loc['CHEM_100002035', '2.5_ci']
log_ci_upper = coef_df.loc['CHEM_100002035', '97.5_ci']

# Calculate the error margins (distance from the estimate to the bounds)
error_lower = log_coef - log_ci_lower
error_upper = log_ci_upper - log_coef

# Create a forest plot for log(OR)
fig, ax = plt.subplots(figsize=(5, 3))
ax.errorbar(log_coef, 1, xerr=[[error_lower], [error_upper]], fmt='o', color='blue', capsize=5)

# Add a vertical reference line at 0 (which corresponds to OR = 1)
ax.axvline(x=0, color='gray', linestyle='--')

# Set labels and title
ax.set_xlabel("log(Odds Ratio)")
ax.set_yticks([])  # Remove y-axis ticks
ax.set_title("log(Odds Ratio) for CHEM_100002035")

# Display the plot
plt.show()

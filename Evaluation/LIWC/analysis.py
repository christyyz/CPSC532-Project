# %%
import pandas as pd
from scipy.stats import ttest_ind

model = 'gpt-personx'
df_female = pd.read_csv(f"{model}-female.csv")
df_male = pd.read_csv(f"{model}-male.csv")

numeric_cols = df_female.select_dtypes(include=['number']).columns
numeric_cols = [col for col in numeric_cols if col in df_male.columns]

results = []

for col in numeric_cols:
    female_vals = df_female[col].dropna()
    male_vals = df_male[col].dropna()

    t_stat, p_val = ttest_ind(female_vals, male_vals, equal_var=False)

    results.append({
        "Feature": col,
        "Female Mean": female_vals.mean(),
        "Male Mean": male_vals.mean(),
        "T-Stat": t_stat,
        "P-Value": p_val
    })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("P-Value")
df_results

# %%
df_results[df_results['P-Value']<0.05].to_csv(f'{model}-LIWC-RESULT.csv', index=False)
# %%

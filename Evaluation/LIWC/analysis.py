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
import pandas as pd

df_bart = pd.read_csv('./results/bart-LIWC-RESULT.csv')
bart = [
    ("swear", "shit, fuckin*, fuck, damn"),
    ("tone_neg", "bad, wrong, too much, hate"),
    ("emo_anger", "hate, mad, angry, frustr*"),
    ("family", "parent*, mother*, father*, baby"),
    ("friend", "friend*, boyfriend*, girlfriend*, dude"),
    ("emo_neg", "bad, hate, hurt, tired")
]
df_liwc_bart = pd.DataFrame(bart, columns=["Category", "Description / Most Frequent Examples"])
df_bart = df_bart.merge(df_liwc_bart, how="right", left_on="Feature", right_on="Category")
df_bart = df_bart[['Category', 'Description / Most Frequent Examples', 'Female Mean', 'Male Mean', 'T-Stat', 'P-Value']]

custom_order = ['family', 'friend', 'swear', 'tone_neg', 'emo_neg', 'emo_anger']
df_bart = df_bart[df_bart['Category'].isin(custom_order)].set_index('Category').loc[custom_order].reset_index()
df_bart.to_csv('./results/BART-RESULT.csv',index=False)
df_bart
# %%
import pandas as pd

df_gpt = pd.read_csv('./results/gpt-LIWC-RESULT.csv')
gpt = [
    ("risk", "secur*, protect*, pain, risk*"),
    ("prosocial", "care, help, thank, please"),
    ("power", "own, order, allow, power"),
    ("comm", "said, say, tell, thank*"),
    ("fulfill", "enough, full, complete, extra"),
    ("tone_pos", "good, well, new, love"),
    ("reward", "opportun*, win, gain*, benefit*"),
    ("curiosity", "scien*, look* for, research*, wonder"),
    ("sexual", "sex, gay, pregnan*, dick"),
    ("moral", "wrong, honor*, deserv*, judge"),
    ("lack", "don’t have, didn’t have, *less, hungry"),
    ("polite", "thank, please, thanks, good morning"),
    ("death", "death*, dead, die, kill"),
    ("money", "business*, pay*, price*, market*"),
    ("home", "home, house, room, bed"),
    ("emo_anger", "hate, mad, angry, frustr*")
]


df_liwc_gpt = pd.DataFrame(gpt, columns=["Category", "Description / Most Frequent Examples"])
df_gpt = df_gpt.merge(df_liwc_gpt, how="right", left_on="Feature", right_on="Category")
df_gpt = df_gpt[['Category', 'Description / Most Frequent Examples', 'Female Mean', 'Male Mean', 'T-Stat', 'P-Value']]
df_gpt
custom_order = ['emo_anger', 'lack', 'power', 'fulfill', 'reward', 'prosocial', 'moral', 'polite', 'money', 'home']
df_gpt = df_gpt[df_gpt['Category'].isin(custom_order)].set_index('Category').loc[custom_order].reset_index()
df_gpt.to_csv('./results/GPT-RESULT.csv',index=False)
df_gpt
# %%

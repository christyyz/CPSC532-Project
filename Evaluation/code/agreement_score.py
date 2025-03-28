#%%
from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

# %%
def agreement_score(llm_model, neutral):
    df = pd.read_csv(f'../{llm_model}_inference_{neutral}_ALL.csv')
    for index, row in df.iterrows():
        inferences = [row['Female Inference'], row['Male Inference'], row[f'{neutral} Inference']]
        # F, M ,U
        embeddings = model.encode(inferences, convert_to_tensor=True)
        sim_female_male = util.cos_sim(embeddings[0], embeddings[1]).item()
        sim_female_unisex = util.cos_sim(embeddings[0], embeddings[2]).item()
        sim_male_unisex = util.cos_sim(embeddings[1], embeddings[2]).item()
        
        df.loc[index, 'Female-Male Similarity'] = sim_female_male
        df.loc[index, f'Female-{neutral} Similarity'] = sim_female_unisex
        df.loc[index, f'Male-{neutral} Similarity'] = sim_male_unisex
        
    df.to_csv(f'../results/{llm_model}_{neutral}_agreement_score_ALL.csv', index=False)
# %%
agreement_score('bart', 'Unisex')
agreement_score('bart', 'PersonX')
# %%
agreement_score('gpt2', 'Unisex')
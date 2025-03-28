#%%
import pandas as pd
import ast

# BART OUTPUT
df_bart_female = pd.read_csv('../comet_inferences_female_2025-03-25_00-06-56.csv')
df_bart_male = pd.read_csv('../comet_inferences_male_2025-03-25_00-40-30.csv')
df_bart_unisex = pd.read_csv('../comet_inferences_unisex_2025-03-25_01-13-38.csv')

def process_df(df):
    for index, row in df.iterrows():
        inference = row['Inference']
        inference_lst = ast.literal_eval(inference)[0]
        first_non_none = next((x for x in inference_lst if x != 'none'), None)
        df.loc[index, 'First Inference'] = first_non_none
        
process_df(df_bart_female)
process_df(df_bart_male)
process_df(df_bart_unisex)
df_bart_female.rename(columns={'First Inference': 'Female Inference'}, inplace=True)
df_bart_male.rename(columns={'First Inference': 'Male Inference'}, inplace=True)
df_bart_unisex.rename(columns={'First Inference': 'Unisex Inference'}, inplace=True)

def get_event_relation(df):
    for index, row in df.iterrows():
        query = row['Query']
        event_relation = '[Name] ' + ' '.join(query.split()[1:])
        df.loc[index, 'Event Relation'] = event_relation
        
get_event_relation(df_bart_female)
get_event_relation(df_bart_male)
get_event_relation(df_bart_unisex)

df_bart_female = df_bart_female[['Event Relation', 'Female Inference']]
df_bart_male = df_bart_male[['Event Relation', 'Male Inference']]
df_bart_unisex = df_bart_unisex[['Event Relation', 'Unisex Inference']]

merged_df = pd.merge(df_bart_female, df_bart_male, on='Event Relation', how='inner')
merged_df = pd.merge(merged_df, df_bart_unisex, on='Event Relation', how='inner')
merged_df.to_csv('../bart_Unisex_inference_ALL.csv', index=False)

#%%
import pandas as pd
import ast

# BART PERSONX
df_bart_female = pd.read_csv('../comet_inferences_female_2025-03-25_00-06-56.csv')
df_bart_male = pd.read_csv('../comet_inferences_male_2025-03-25_00-40-30.csv')
df_bart_personx = pd.read_csv('../comet_inferences_PersonX_2025-03-28_18-59-02.csv')

def process_df(df):
    for index, row in df.iterrows():
        inference = row['Inference']
        inference_lst = ast.literal_eval(inference)[0]
        first_non_none = next((x for x in inference_lst if x != 'none'), None)
        df.loc[index, 'First Inference'] = first_non_none
        
process_df(df_bart_female)
process_df(df_bart_male)
process_df(df_bart_personx)
df_bart_female.rename(columns={'First Inference': 'Female Inference'}, inplace=True)
df_bart_male.rename(columns={'First Inference': 'Male Inference'}, inplace=True)
df_bart_personx.rename(columns={'First Inference': 'PersonX Inference'}, inplace=True)

def get_event_relation(df):
    for index, row in df.iterrows():
        query = row['Query']
        event_relation = '[Name] ' + ' '.join(query.split()[1:])
        df.loc[index, 'Event Relation'] = event_relation
        
get_event_relation(df_bart_female)
get_event_relation(df_bart_male)
get_event_relation(df_bart_personx)

df_bart_female = df_bart_female[['Event Relation', 'Female Inference']]
df_bart_male = df_bart_male[['Event Relation', 'Male Inference']]
df_bart_personx = df_bart_personx[['Event Relation', 'PersonX Inference']]

merged_df = pd.merge(df_bart_female, df_bart_male, on='Event Relation', how='inner')
merged_df = pd.merge(merged_df, df_bart_personx, on='Event Relation', how='inner')
merged_df.to_csv('../bart_inference_PersonX_ALL.csv', index=False)

# %%
import pandas as pd

# GPT2 OUTPUT
df_gpt2_female = pd.read_csv('../Copy of integrated_female.csv')
df_gpt2_male = pd.read_csv('../Copy of integrated_male.csv')
df_gpt2_unisex = pd.read_csv('../Copy of integrated_unisex.csv')

def process_df(df):
    for index, row in df.iterrows():
        inference = row['generations']
        if type(inference) == float: # need updated after filling in missing data
            df.loc[index, 'First Inference'] = 'none'
            print(index)
            continue
        inference_lst = inference.split(', ')
        first_non_none = next((x for x in inference_lst if x != 'none'), None)
        df.loc[index, 'First Inference'] = first_non_none

process_df(df_gpt2_female)
process_df(df_gpt2_male)
process_df(df_gpt2_unisex)
df_gpt2_female.rename(columns={'First Inference': 'Female Inference'}, inplace=True)
df_gpt2_male.rename(columns={'First Inference': 'Male Inference'}, inplace=True)
df_gpt2_unisex.rename(columns={'First Inference': 'Unisex Inference'}, inplace=True)

import re
def get_event_relation(df):
    for index, row in df.iterrows():
        query = row['source']
        event_relation = '[Name] ' + ' '.join(query.split()[1:])
        cleaned_event_relation = re.sub(r'\s+', ' ', event_relation)
        df.loc[index, 'Event Relation'] = cleaned_event_relation.strip()
      
get_event_relation(df_gpt2_female)
get_event_relation(df_gpt2_male)
get_event_relation(df_gpt2_unisex)

df_gpt2_female = df_gpt2_female[['Event Relation', 'Female Inference']]
df_gpt2_male = df_gpt2_male[['Event Relation', 'Male Inference']]
df_gpt2_unisex = df_gpt2_unisex[['Event Relation', 'Unisex Inference']] 

merged_df = pd.merge(df_gpt2_female, df_gpt2_male, on='Event Relation', how='inner')
merged_df = pd.merge(merged_df, df_gpt2_unisex, on='Event Relation', how='inner')
merged_df.to_csv('../gpt2_inference_Unisex_ALL.csv',index=False)
# %%
import pandas as pd

# GPT2 PERSONX
df_gpt2_female = pd.read_csv('../Copy of integrated_female.csv')
df_gpt2_male = pd.read_csv('../Copy of integrated_male.csv')
df_gpt2_personX = pd.read_csv('../Copy of extracted_PersonX.csv')

def process_df(df, column):
    for index, row in df.iterrows():
        inference = row[column]
        if type(inference) == float: # need updated after filling in missing data
            df.loc[index, 'First Inference'] = 'none'
            print(index)
            continue
        inference_lst = inference.split(', ')
        first_non_none = next((x for x in inference_lst if x != 'none'), None)
        df.loc[index, 'First Inference'] = first_non_none

process_df(df_gpt2_female, 'generations')
process_df(df_gpt2_male, 'generations')
process_df(df_gpt2_personX, 'extracted_generations')
df_gpt2_female.rename(columns={'First Inference': 'Female Inference'}, inplace=True)
df_gpt2_male.rename(columns={'First Inference': 'Male Inference'}, inplace=True)
df_gpt2_personX.rename(columns={'First Inference': 'PersonX Inference'}, inplace=True)

import re
def get_event_relation(df):
    for index, row in df.iterrows():
        query = row['source']
        event_relation = '[Name] ' + ' '.join(query.split()[1:])
        cleaned_event_relation = re.sub(r'\s+', ' ', event_relation)
        df.loc[index, 'Event Relation'] = cleaned_event_relation.strip()
      
get_event_relation(df_gpt2_female)
get_event_relation(df_gpt2_male)
get_event_relation(df_gpt2_personX)

df_gpt2_female = df_gpt2_female[['Event Relation', 'Female Inference']]
df_gpt2_male = df_gpt2_male[['Event Relation', 'Male Inference']]
df_gpt2_personX = df_gpt2_personX[['Event Relation', 'PersonX Inference']] 

merged_df = pd.merge(df_gpt2_female, df_gpt2_male, on='Event Relation', how='inner')
merged_df = pd.merge(merged_df, df_gpt2_personX, on='Event Relation', how='inner')
merged_df.to_csv('../gpt2_inference_PersonX_ALL.csv',index=False)

# %%
###Round all output files
import pandas as pd
import os

def round_csv(filepath):
    df = pd.read_csv(filepath)
    df = df.round(3)
    df.to_csv(filepath, index=False)

def round_all_csvs_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            round_csv(filepath)

#%%   
round_all_csvs_in_folder('../results/bart_personX/')
round_all_csvs_in_folder('../results/bart_unisex/')
round_all_csvs_in_folder('../results/gpt2_unisex/')
round_all_csvs_in_folder('../results/gpt2_personX/')
# %%

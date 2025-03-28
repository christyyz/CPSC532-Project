#%%
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('stopwords')
nltk.download('vader_lexicon')

#%%
sia = SentimentIntensityAnalyzer()
vader_words = set(sia.lexicon.keys())

stop_words = set(stopwords.words('english'))
ignore_words = stop_words.union({'he', 'she', 'his', 'her', 'PersonX'})
name_list_female = set(pd.read_csv('../../Dataset/female_names.csv')['Female Name'].tolist())
name_list_male = set(pd.read_csv('../../Dataset/male_names.csv')['Male Name'].tolist())
name_list_unisex = set(pd.read_csv('../../Dataset/unisex_names.csv')['Name'].tolist())
ignore_words = ignore_words.union(name_list_female).union(name_list_male).union(name_list_unisex)

#%%
def lexical_analysis(llm_model, neutral):
    def tokenize_phrase(text):
        if pd.isna(text):
            return []
        return re.findall(r'\b\w+\b', str(text).lower())

    df = pd.read_csv(f"../{llm_model}_inference_{neutral}_ALL.csv")

    df['female_tokens'] = df['Female Inference'].apply(tokenize_phrase)
    df['male_tokens'] = df['Male Inference'].apply(tokenize_phrase)
    df['unisex_tokens'] = df[f'{neutral} Inference'].apply(tokenize_phrase)

    female_words = [w for tokens in df['female_tokens'] for w in tokens]
    male_words = [w for tokens in df['male_tokens'] for w in tokens]
    unisex_words = [w for tokens in df['unisex_tokens'] for w in tokens]

    female_freq = Counter(female_words)
    male_freq = Counter(male_words)
    unisex_freq = Counter(unisex_words)

    biased_words = []

    for word in set(female_freq.keys()).union(male_freq.keys()).union(unisex_freq.keys()):
        if word in ignore_words or word not in vader_words:
            continue

        f = female_freq[word]
        m = male_freq[word]
        u = unisex_freq[word]
        if f + m + u < 5: 
            continue
        
        f_m_ratio = (f + 1) / (m + 1)
        sentiment = sia.polarity_scores(word)['compound']
        biased_words.append((word, f, m, u, f_m_ratio, sentiment))

    biased_df = pd.DataFrame(biased_words, columns=["word", "female_count", "male_count", f"{neutral}_count", "f_m_ratio", "sentiment"])

    biased_df.to_csv(f"../results/{llm_model}_{neutral}_lexical_analysis_ALL.csv", index=False)

    biased_df[biased_df['f_m_ratio'] < 0.33].to_csv(f'../results/{llm_model}_{neutral}_lexical_analysis_male_biased_words.csv', index=False)
    biased_df[biased_df['f_m_ratio'] > 3].to_csv(f'../results/{llm_model}_{neutral}_lexical_analysis_female_biased_words.csv', index=False)
# %%
lexical_analysis('bart', 'Unisex')
lexical_analysis('bart', 'PersonX')
lexical_analysis('gpt2', 'Unisex')
lexical_analysis('gpt2', 'PersonX')
# %%

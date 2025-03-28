#%%
import pandas as pd
import nltk
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def sentiment_analysis(df, gender):
    sia = SentimentIntensityAnalyzer()
    for index, row in df.iterrows():
        inference = row[f'{gender} Inference']
        sentiments = sia.polarity_scores(inference)
        df.loc[index, f'{gender} Sentiment: Negative'] = sentiments['neg']
        df.loc[index, f'{gender} Sentiment: Neutral'] = sentiments['neu']
        df.loc[index, f'{gender} Sentiment: Positive'] = sentiments['pos']
        df.loc[index, f'{gender} Sentiment: Compound'] = sentiments['compound']
    
# %%
df = pd.read_csv('../bart_inference_Unisex_ALL.csv')
sentiment_analysis(df, 'Female')
sentiment_analysis(df, 'Male')
sentiment_analysis(df, 'Unisex')
df.to_csv('../results/bart_Unisex_sentiment_analysis_ALL.csv', index=False)
# %%
df = pd.read_csv('../bart_inference_PersonX_ALL.csv')
sentiment_analysis(df, 'Female')
sentiment_analysis(df, 'Male')
sentiment_analysis(df, 'PersonX')
df.to_csv('../results/bart_PersonX_sentiment_analysis_ALL.csv', index=False)
# %%
df = pd.read_csv('../gpt2_inference_Unisex_ALL.csv')
df = df.astype(str)
sentiment_analysis(df, 'Female')
sentiment_analysis(df, 'Male')
sentiment_analysis(df, 'Unisex')
df.to_csv('../results/gpt2_Unisex_sentiment_analysis_ALL.csv', index=False)
# %%
df = pd.read_csv('../gpt2_inference_PersonX_ALL.csv')
df = df.astype(str)
sentiment_analysis(df, 'Female')
sentiment_analysis(df, 'Male')
sentiment_analysis(df, 'PersonX')
df.to_csv('../results/gpt2_PersonX_sentiment_analysis_ALL.csv', index=False)
# %%

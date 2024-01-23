import nltk
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator


nltk.download('vader_lexicon')
excel_file = 'C:\\Users\\ccbbb\\Downloads\\base_coment.xlsx'
df = pd.read_excel(excel_file)
sia = SentimentIntensityAnalyzer()

translator = GoogleTranslator(source='pt', target='en')

def translate_text(text):
    if isinstance(text, str) and len(text) <= 5000:
        return translator.translate(text)
    else:
        return text

df['frase_em_ingles'] = df['Comentário'].apply(translate_text)
df['dados_nltk'] = df['frase_em_ingles'].apply(lambda x: sia.polarity_scores(str(x)))
df['compound'] = df['dados_nltk'].apply(lambda x: x['compound'])
df['status'] = np.where(df['compound'] > 0, 'Positivo', np.where(df['compound'] == 0, 'Neutro', 'Negativo'))



df.to_excel(f'C:\\Users\\ccbbb\\Downloads\\resultado.xlsx', index=False)
print(df[['Comentário', 'frase_em_ingles']])

print(df.iloc[:, [0, 1]])
print("fim")
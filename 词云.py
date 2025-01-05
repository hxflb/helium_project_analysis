import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')

csv_file = 'commit_history.csv'
df = pd.read_csv(csv_file, header=None ,on_bad_lines='skip')
descriptions = df.iloc[:, 3]

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

preprocessed_descriptions = descriptions.apply(preprocess_text)
all_text = ' '.join(preprocessed_descriptions)
stop_words = set(stopwords.words('english'))
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
output_file = 'Most_mentioned.png'
plt.savefig(output_file, bbox_inches='tight')
plt.show()
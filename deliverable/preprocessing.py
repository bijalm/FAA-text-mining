import numpy as np
import re
import nltk
import string

from nltk.corpus import stopwords
# Checking nltk package dependencies
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Stopword removal function
stop_words = stopwords.words('english')

def remove_stop_words(narrative):
  return ' '.join([word for word in narrative.split() if word not in stop_words and len(word) > 1])

def clean_narratives(narratives):
    # Remove all tags from narratives
    narratives = [re.sub(r'<(.+?)>','',narrative) for narrative in narratives]

    # Stripping all punctuation from narratives
    narratives = [narrative.translate(str.maketrans('','',string.punctuation)) for narrative in narratives]

    # Remove all stopwords from narratives
    narratives = list(map(remove_stop_words,narratives))
    return np.array(narratives)

def classifier(x, colNames):
  labels = []

  for col, val in enumerate(x):
    if val == 1 and colNames[col] != 'INDEX' and colNames[col] != 'REPORT DATE' and colNames[col] != 'Unnamed: 0':
      labels.append(colNames[col])
  return labels
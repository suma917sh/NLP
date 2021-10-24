#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##### Extract reviews of any product from ecommerce website like amazon 


# In[ ]:


##### Perform emotion mining


# In[7]:


#BeautifulSoup IS PYTHON LIBRARY USED TO PULL DATA FROM HTML & XML FILES
from bs4 import BeautifulSoup as bs
import requests


# READING THE DATA

# In[8]:


bt='https://www.amazon.in'
ul='https://www.amazon.in/Apple-MacBook-Air-13-3-inch-MQD32HN/product-reviews/B073Q5R6VR/ref=cm_cr_getr_d_paging_btm_next_30?ie=UTF8&reviewerType=all_reviews'


# In[9]:


#LIST TO STORE NAME OF CUSTOMERS
cust_name = []   
review_title = []
rate = []
review_content = []


# In[10]:


tt = 0
while tt == 0:
    page = requests.get(ul)
    while page.ok == False:
        page = requests.get(ul)
   

    soup = bs(page.content,'html.parser')
    soup.prettify()  #Prettify() function in BeautifulSoup 
    
    names = soup.find_all('span', class_='a-profile-name')
    names.pop(0)
    names.pop(0)
    
    for i in range(0,len(names)):
        cust_name.append(names[i].get_text())
        
    title = soup.find_all("a",{"data-hook":"review-title"})
    for i in range(0,len(title)):
        review_title.append(title[i].get_text())

    rating = soup.find_all('i',class_='review-rating')
    rating.pop(0)
    rating.pop(0)
    for i in range(0,len(rating)):
        rate.append(rating[i].get_text())

    review = soup.find_all("span",{"data-hook":"review-body"})
    for i in range(0,len(review)):
        review_content.append(review[i].get_text())
        
    try:
        for div in soup.findAll('li', attrs={'class':'a-last'}):
            A = div.find('a')['href']
        ul = bt + A
    except:
        break


# In[5]:


len(cust_name)


# In[6]:


len(review_title)


# In[7]:


len(review_content)


# In[8]:


len(rate)


# In[9]:


review_title[:] = [titles.lstrip('\n') for titles in review_title]

review_title[:] = [titles.rstrip('\n') for titles in review_title]

review_content[:] = [titles.lstrip('\n') for titles in review_content]

review_content[:] = [titles.rstrip('\n') for titles in review_content]


# In[11]:


#IMPORTING NECESSARY LIBRARIES
get_ipython().system('pip install -U textblob')
get_ipython().system('python -m textblob.download_corpora')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import word_tokenize
from textblob import TextBlob, Word, Blobber
import wordcloud
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
nltk.download('stopwords')


# In[11]:


df = pd.DataFrame()


# In[12]:


df['Customer Name'] = cust_name
df['Review Title'] = review_title
df['Rating'] = rate
df['Reviews'] = review_content


# In[13]:


df.head(10)


# In[14]:


df.to_csv(r'E:fill.csv',index = True)


# In[15]:


data = pd.read_csv("E:fill.csv",index_col=[0])


# In[16]:


data.dtypes


# In[17]:


data['Rating'] = [titles.rstrip(' out of 5 stars') for titles in data['Rating']]


# In[18]:


data['Rating']


# In[19]:


data['Rating'].value_counts(normalize=True)*100


# In[20]:


ratings=data.groupby(['Rating']).count()
ratings


# VISUALIZATION

# In[21]:


plt.figure(figsize=(12,8))
data['Rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Rating')
plt.xlabel('Rating')
plt.ylabel('Count')


# In[22]:


data.Rating.hist()
data.Rating.hist(bins=10)
plt.xlabel('Rating')
plt.ylabel('Count')


# In[23]:


data.iloc[:,[3]]


# In[24]:


Reviews=data.iloc[:,[3]]


# In[25]:


Reviews.shape


# In[26]:


Reviews.describe()


# In[27]:


Reviews.dtypes


# In[28]:


#DROP CUSTOMERNAME AND REVIEW TITLE COLUMN 
data.drop(["Customer Name","Review Title"],axis=1,inplace=True)
data.head()


# In[32]:


data.Reviews.isna().sum()


# In[33]:


data['Reviews']=data['Reviews'].fillna(" ")


# In[34]:


data.Reviews.isna().sum()


# In[35]:


#CLEANING THE DATA
data['Reviews']= data['Reviews'].apply(lambda x: " ".join(word.lower() for word in x.split()))


# In[36]:


#REMOVE PUNCTUATIONS
import string
data['Reviews']=data['Reviews'].apply(lambda x:''.join([i for i in x  if i not in string.punctuation]))


# In[37]:


#REMOVE NUMBERS
data['Reviews']=data['Reviews'].str.replace('[0-9]','')


# In[38]:


#REMOVE STOPWORDS
from nltk.corpus import stopwords


# In[39]:


stop_words=stopwords.words('english')


# In[40]:


data['Reviews']=data['Reviews'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))


# In[41]:


data.head(5)


# In[42]:


from textblob import Word
data['Reviews']= data['Reviews'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
import re
pattern = r"((?<=^)|(?<= )).((?=$)|(?= ))"
data['Reviews']= data['Reviews'].apply(lambda x:(re.sub(pattern, '',x).strip()))


# In[43]:


data['Reviews'].head()


# #### FEATURE EXTRACTION

# In[44]:


from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(data['Reviews'])
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(df)


# #### TF-IDF VECTORIZER

# In[45]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
TFIDF=tfidf.fit_transform(data['Reviews'])
print(TFIDF)


# In[46]:


#GENERATE WORDCLOUD
Review_wordcloud = ' '.join(data['Reviews'])
Q_wordcloud=WordCloud(
                    background_color='white',
                    width=2000,
                    height=2000
                   ).generate(Review_wordcloud)
fig = plt.figure(figsize = (10, 10))
plt.axis('on')
plt.imshow(Q_wordcloud)


# ##### REMOVING PUNCTUATION 

# In[47]:


data['Reviews'] = data['Reviews'].str.replace('[^\w\s]','')
data['Reviews'].head()


# ##### REMOVAL OF COMMON WORDS

# In[48]:


freq = pd.Series(' '.join(data['Reviews']).split()).value_counts()[:10]
freq


# In[49]:


data['Reviews'] = data['Reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
data['Reviews'].head()


# ##### REMOVAL OF RARE WORDS 

# In[50]:


freq = pd.Series(' '.join(data['Reviews']).split()).value_counts()[-10:]
freq


# In[51]:


#TEXTBLOB FOR PROCESSING TEXTUAL DATA
from textblob import TextBlob
data['Reviews'][:10].apply(lambda x: str(TextBlob(x).correct()))


# ##### TOKENIZATION 

# In[52]:


TextBlob(data['Reviews'][0]).words


# In[53]:


TextBlob(data['Reviews'][1]).words


# ##### STEMMING 

# REMOVAL OF SUFFIXES

# In[54]:


from nltk.stem import PorterStemmer
st = PorterStemmer()
data['Reviews'][:10].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# ##### LEMMATIZATION

# In[55]:


from textblob import Word
data['Reviews'] = data['Reviews'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['Reviews'].head()


# ##### ADVANCE TEXT PROCESSING

# In[56]:


#NGRAMS
TextBlob(data['Reviews'][0]).ngrams(2)


# ##### TERM FREQUENCY (TF)

# In[57]:


tf1 = (data['Reviews'][1:10]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1


# ##### INVERSE DOCUMENT FREQUENCY (IDF)

# In[58]:


for i,word in enumerate(tf1['words']):
    tf1.loc[i, 'idf'] = np.log(data.shape[0]/(len(data[data['Reviews'].str.contains(word)])))


# In[59]:


tf1


# More the value of IDF,more unique is the word.

# In[60]:


tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1


# ##### TF-IDF VECTORIZER 

# In[61]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
data_vect = tfidf.fit_transform(data['Reviews'])

data_vect


# #### BAG OF WORDS(BOW)
# For implementation, sklearn provides a separate function 

# In[62]:


from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
data_bow = bow.fit_transform(data['Reviews'])
data_bow


# ##### SENTIMENT ANALYSIS 

# In[63]:


data['Reviews'][:10].apply(lambda x: TextBlob(x).sentiment)


# VALUE NEARER TO 1 CONSIDERED A POSITIVE SENTIMENT AND NEARER TO -1 CONSIDERED A NEGATIVE SENTIMENT

# In[64]:


data['sentiment'] = data['Reviews'].apply(lambda x: TextBlob(x).sentiment[0] )
data[['Reviews','sentiment']].head()


# In[13]:


get_ipython().system('pip install gensim')
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# In[95]:


from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'negative-words.txt'
word2vec_output_file = 'positive-words.txtpd.read_csv'


# In[14]:


#glove2word2vec(glove_input_file, word2vec_output_file)


# In[15]:


import collections
from collections import Counter
import nltk
nltk.download('punkt')


# In[73]:


from textblob import TextBlob
data['polarity'] = data['Reviews'].apply(lambda x: TextBlob(x).sentiment[0])
data[['Reviews','polarity']].head(5)


# In[74]:


data[data.polarity>0].head(5)


# In[75]:


def sent_type(text): 
    for i in (text):
        if i>0:
            print('positive')
        elif i==0:
            print('neutral')
        else:
            print('negative')


# In[76]:


sent_type(data['polarity'])


# In[77]:


data["category"]=data['polarity']


# In[78]:


data.loc[data.category > 0,'category']="Positive"
data.loc[data.category !='Positive','category']="Negative"


# In[79]:


data["category"]=data["category"].astype('category')
data.dtypes


# VISUALIZATION

# In[80]:


sns.countplot(x='category',data=data,palette='hls')


# WORDCLOUD

# In[81]:


positive_reviews= data[data.category=='Positive']
negative_reviews= data[data.category=='Negative']
positive_reviews_text=" ".join(positive_reviews.Reviews.to_numpy().tolist())
negative_reviews_text=" ".join(negative_reviews.Reviews.to_numpy().tolist())
positive_reviews_cloud=WordCloud(background_color='black',max_words=150).generate(positive_reviews_text)
negative_reviews_cloud=WordCloud(background_color='black',max_words=150).generate(negative_reviews_text)
plt.imshow(positive_reviews_cloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0) 
plt.show()
plt.imshow(negative_reviews_cloud,interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0) 
plt.show()


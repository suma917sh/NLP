#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Importing necessary libraries
import pandas as pd
import tweepy
import re


# In[17]:


# Using the consumer keys and access key to extract the tweets from twitter
consumer_key = "YjpUMq8F0Vc15DrmDT1dVNHYO"
consumer_secret = "hqqKizfetGcQGDziLVHguaBRMKqujnC98GRzp4BaVWIy2YdtQi"
access_key = "2821184361-nI4ZUrqAhJ3bQq0e15rUuFMXbHTK3VuJvIoVWHp"
access_secret = "ltXVMn0Iqn7GJnJwFXUDDIxgB7Zw4ne0uL8B1qzlcjj47"


# In[18]:


#Tweet extraction
alltweets = []


# In[19]:


# Defining the function to get the tweets
def get_all_tweets(screen_name):
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    alltweets.extend(new_tweets)
    
    oldest = alltweets[-1].id - 1
    while len(new_tweets)>0:
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        #save most recent tweets
        alltweets.extend(new_tweets)
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print ("...%s tweets downloaded so far" % (len(alltweets)))                # tweet.get('user', {}).get('location', {})
 
    outtweets = [[tweet.created_at,tweet.entities["hashtags"],tweet.entities["user_mentions"],tweet.favorite_count,
                  tweet.geo,tweet.id_str,tweet.lang,tweet.place,tweet.retweet_count,tweet.retweeted,tweet.source,tweet.text,
                  tweet._json["user"]["location"],tweet._json["user"]["name"],tweet._json["user"]["time_zone"],
                  tweet._json["user"]["utc_offset"]] for tweet in alltweets]
     
    tweets_df = pd.DataFrame(columns = ["time","hashtags","user_mentions","favorite_count",
                                    "geo","id_str","lang","place","retweet_count","retweeted","source",
                                    "text","location","name","time_zone","utc_offset"])
    tweets_df["time"]  = pd.Series([str(i[0]) for i in outtweets])
    tweets_df["hashtags"] = pd.Series([str(i[1]) for i in outtweets])
    tweets_df["user_mentions"] = pd.Series([str(i[2]) for i in outtweets])
    tweets_df["favorite_count"] = pd.Series([str(i[3]) for i in outtweets])
    tweets_df["geo"] = pd.Series([str(i[4]) for i in outtweets])
    tweets_df["id_str"] = pd.Series([str(i[5]) for i in outtweets])
    tweets_df["lang"] = pd.Series([str(i[6]) for i in outtweets])
    tweets_df["place"] = pd.Series([str(i[7]) for i in outtweets])
    tweets_df["retweet_count"] = pd.Series([str(i[8]) for i in outtweets])
    tweets_df["retweeted"] = pd.Series([str(i[9]) for i in outtweets])
    tweets_df["source"] = pd.Series([str(i[10]) for i in outtweets])
    tweets_df["text"] = pd.Series([str(i[11]) for i in outtweets])
    tweets_df["location"] = pd.Series([str(i[12]) for i in outtweets])
    tweets_df["name"] = pd.Series([str(i[13]) for i in outtweets])
    tweets_df["time_zone"] = pd.Series([str(i[14]) for i in outtweets])
    tweets_df["utc_offset"] = pd.Series([str(i[15]) for i in outtweets])
    tweets_df.to_csv(screen_name+"_tweets.csv")
    return tweets_df


# In[20]:


# Extracting tweets of "SRK"
srk_tweets = get_all_tweets("iamsrk")


# In[21]:


# Checking the extracted tweets
srk_tweets


# In[22]:


# Extracting only the 'text' section of the tweet
srk_tweets["text"]


# In[23]:


# Combining all the tweets into a single paragraph
tweet_df = " ".join(srk_tweets["text"])
tweet_df


# In[24]:


#Text processing
# Removing all the http links from the tweets
tweet_df = re.sub(r"http\S+", "", tweet_df)
tweet_df


# In[25]:


# Removing all the puntuations, numbers and lowering the case of  letters
tweets = re.sub("[^A-Za-z" "]+"," ",tweet_df).lower()
tweets = re.sub("[0-9" "]+"," ",tweets)
tweets


# In[26]:


# Splitting the paragraph into words
tweet_words = tweets.split(" ")
tweet_words
len(tweet_words)


# In[27]:


# Importing the stop words
with open(r"C:\\Users\\hp\\Downloads\\stop.txt","r") as sw:
    stop = sw.read()
stop


# In[28]:


# Removing all the stop words from the tweets
tweets_final = [w for w in tweet_words if not w in stop]
tweets_final
len(tweets_final)


# In[29]:


# Joining all the tweet words into one paragraph
tweets_final_string = " ".join(tweets_final)
tweets_final_string


# In[30]:


#Sentimental analysis
import numpy as np
import matplotlib.pyplot as plt


# In[36]:


from wordcloud import WordCloud

# Plotting a wordcloud with all the unique words in the tweets
wordcloud_tweet = WordCloud(
                    background_color="black",
                    width=1800,
                    height=1400,
                    ).generate(tweets_final_string)
plt.imshow(wordcloud_tweet)


# In[ ]:


# List of all the unique words in the reviews
tweet_unique = list(set("".join(tweets_final_string).split(" ")))
tweet_unique
len(tweet_unique)


# In[37]:


# Importing the positive words
with open(r"C:\\Users\\hp\\Downloads\\positive-words.txt","r") as pos:
    poswords= pos.read().split("\n")
poswords = poswords[36:]


# In[38]:


# Importing the negative words
with open(r"C:\\Users\\hp\\Downloads\\negative-words.txt","r") as neg:
    negwords = neg.read().split("\n")

negwords = negwords[37:]


# In[39]:


# Joining all the negative words into a paragraph
tweet_neg = " ".join([w for w in tweets_final if w in negwords])
tweet_neg
len(tweet_neg)


# In[40]:


# Builiding wordcloud of negative words
wordcloud_neg = WordCloud(
                    background_color="black",
                    width=1800,
                    height=1400,
                    ).generate(tweet_neg)
plt.imshow(wordcloud_neg)


# In[41]:


# Joining all the positive words into a paragraph
tweet_pos= " ".join([w for w in tweets_final if w in poswords])
tweet_pos
len(tweet_pos)


# In[42]:


# Builiding wordcloud of positive words
wordcloud_pos = WordCloud(
                background_color="black",
                width=1800,
                height=1400,
                ).generate(tweet_pos)
plt.imshow(wordcloud_pos)


# In[43]:


#Most frequent positive words are "Love", "Happy", "beautiful","good","work","fun"
#Most frequent negative words are "miss","damn","bad","missed"
#and no of positive words are higher(15332) than neg words(2588)


# In[ ]:





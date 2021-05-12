#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda list


# In[3]:


from transformers import PegasusTokenizer , PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests


# In[4]:


# 2. Setup Model
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)


# In[5]:


type(tokenizer)


# In[6]:


##scraping data from url using bs4

url = "https://au.finance.yahoo.com/news/china-restricting-tesla-use-uncovers-a-significant-challenge-for-elon-musk-expert-161921664.html"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
paragraphs = soup.find_all('p')


# In[7]:


paragraphs[:10]


# In[10]:


#creating an article using the words from paragraphs list
text = [paragraph.text for paragraph in paragraphs]
words = ' '.join(text).split(' ')[:400]
ARTICLE = ' '.join(words)


# In[31]:


ARTICLE


# In[39]:


#encoding the Article for text generation using transformers
input_ids = tokenizer.encode(ARTICLE, return_tensors='pt')
output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
summary = tokenizer.decode(output[0], skip_special_tokens=True)


# In[40]:


summary


# In[41]:


input_ids.shape


# In[37]:


output[0]


# # Building a news and sentiment pipeline

# In[59]:


monitored_tickers = ['ETH','DOGE','GME', 'TSLA', 'BTC']


# ## search and scrape stock news from yahoo finance

# In[60]:


def search_for_stock_news_urls(ticker):
    search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    hrefs = [link['href'] for link in atags]
    return hrefs


# In[61]:


raw_urls = {ticker:search_for_stock_news_urls(ticker) for ticker in monitored_tickers}
raw_urls


# In[62]:


len(raw_urls['DOGE'])


# In[63]:


import re


# In[64]:


unwanted_key = ['terms','privacy', 'accounts', 'support', 'preferences', 'maps']


# In[65]:


def strip_unwanted_urls(urls, exclude_list):
    val = []
    for url in urls: 
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))


# In[66]:


cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker], unwanted_key) for ticker in monitored_tickers}
cleaned_urls


# ## Search and scrape cleaned urls

# In[67]:


def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs: 
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:350]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES


# In[68]:


articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}
articles


# In[78]:


len(articles['GME'])


# ## Summarizing all articles now
# 

# In[79]:


def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors='pt')
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries


# In[80]:


summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}
summaries


# In[82]:


articles['ETH'][5]


# ## Applying sentiment analysis to the summaries

# In[83]:


from transformers import pipeline
sentiment = pipeline('sentiment-analysis')


# In[94]:


summaries['DOGE'][9]


# In[95]:


sentiment(summaries['DOGE'][5])


# In[98]:


scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers}
scores


# In[99]:


def create_output_array(summaries, scores, urls):
    output = []
    for ticker in monitored_tickers:
        for counter in range(len(summaries[ticker])):
            output_this = [
                ticker,
                summaries[ticker][counter],
                scores[ticker][counter]['label'],
                scores[ticker][counter]['score'],
                urls[ticker][counter]
            ]
            output.append(output_this)
    return output


# In[100]:


final_output = create_output_array(summaries, scores, cleaned_urls)
final_output


# In[101]:


import csv
with open('assetsummaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)


# In[ ]:





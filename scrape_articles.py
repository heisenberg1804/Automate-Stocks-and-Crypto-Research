from bs4 import BeautifulSoup
import requests
import re

#function to search on google and scrape stock ticker based urls form yahoofinance
def search_for_stock_news_urls(ticker):
    search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    a_tags = soup.find_all('a')
    hrefs = [link['href'] for link in a_tags]
    return hrefs

x = input("\n enter the number of ticker summaries you want: ")
user_tickers = [input("\n Enter stock ticker(s): ") for i in range(len(x))] 

raw_urls = {ticker:search_for_stock_news_urls(ticker) for ticker in user_tickers}

#scrape urls which don't have the unwanted stings
unwanted_key = ['terms','privacy', 'accounts', 'support', 'preferences', 'maps']

#function to strip unwanted urls like support , preferences, maps etc
def strip_unwanted_urls(urls, exclude_list):
    val = []
    for url in urls: 
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker], unwanted_key) for ticker in user_tickers}

#function for scraping parapgraphs and creating articles from the scraped URLs
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

articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in user_tickers}



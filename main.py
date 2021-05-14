from model import model, tokenizer, sentiment 
from scrape_articles  import user_tickers, articles, cleaned_urls

#Summarizing all articles now

def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors='pt')
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries


summaries = {ticker:summarize(articles[ticker]) for ticker in user_tickers}

#using pretrained sentiment pipeline and applying it on the summaries
scores = {ticker:sentiment(summaries[ticker]) for ticker in user_tickers}



def create_output_array(summaries, scores, urls):
    output = []
    for ticker in user_tickers:
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



final_output = create_output_array(summaries, scores, cleaned_urls)


import csv
with open('assetsummaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)





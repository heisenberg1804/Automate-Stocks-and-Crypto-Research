from transformers import PegasusTokenizer , PegasusForConditionalGeneration


# 2. Setup Model
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

from transformers import pipeline
sentiment = pipeline('sentiment-analysis')


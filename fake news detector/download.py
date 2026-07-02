#run this to download the model and trained weights for the very first time

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "mrm8488/bert-tiny-finetuned-fake-news-detection"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer.save_pretrained("./fake_news_model/")
model.save_pretrained("./fake_news_model/")


# Sample change
from transformers import pipeline
import json
sentiment_pipeline = pipeline("sentiment-analysis")
data = []

with open("./output/sample2.json", encoding='utf-8') as f: 
    json_val = json.load(f)

print(type(json_val))

for i in range(10):
    data.append(json_val[i]['text'])

print(sentiment_pipeline(data))
from flask import Flask, render_template, request
import torch.nn.functional as F 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model.predict import predict_sentiment

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained('model/saved_model/tokenizer')
model = AutoModelForSequenceClassification.from_pretrained('model/saved_model/distilbert', num_labels=2)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sentiment, probability = predict_sentiment(model, tokenizer, text)
    return render_template('result.html', sentiment=str(sentiment), probability=str(probability), text=text)

if __name__ == '__main__':
    app.run(debug=True)

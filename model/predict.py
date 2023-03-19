import torch.nn.functional as F 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = DistilBertTokenizer.from_pretrained('model/saved_model/tokenizerf')
# model = DistilBertForSequenceClassification.from_pretrained('model/saved_model/distilbert')

# Define a function to make predictions
def predict_sentiment(model, tokenizer, text):
    tokenized_segments = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokenized_segments_input_ids, tokenized_segments_attention_mask = tokenized_segments.input_ids, tokenized_segments.attention_mask
    model_predictions = F.softmax(model(input_ids=tokenized_segments_input_ids, attention_mask=tokenized_segments_attention_mask)['logits'], dim=1)
    pos_prob, neg_prob = model_predictions[0][1].item(), model_predictions[0][0].item()
    sentiment = 'Positive' if pos_prob >= neg_prob else 'Negative'
    probability = round(pos_prob,2) if pos_prob >= neg_prob else round(neg_prob,2)
    return  sentiment, probability

if __name__ == '__main__':
    # Test the predict_sentiment function
    print(predict_sentiment("I love this movie!"))
    print(predict_sentiment("I hate this movie."))

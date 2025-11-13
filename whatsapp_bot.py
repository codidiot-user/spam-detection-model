from flask import Flask, request
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Load your model (same as app.py)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "logesh1962/sms-spam-detector"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

@app.route("/whatsapp", methods=['POST'])
def whatsapp():
    incoming_msg = request.values.get('Body', '').strip().lower()

    if not incoming_msg:
        resp = MessagingResponse()
        resp.message("Send a message to check if it's spam! E.g., 'Win free iPhone!'")
        return str(resp)

    # Your spam detection logic (exact from app.py)
    inputs = tokenizer(incoming_msg, return_tensors='pt', truncation=True, padding=True, max_length=96).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prob_spam = torch.softmax(outputs.logits, dim=-1)[0][1].item()
    label = "Spam/Fake" if prob_spam > 0.3 else "Legitimate"

    if label == "Spam/Fake":
        reply = f"🚨 SPAM ({prob_spam:.0%})! Avoid links & replies.\nWeb app: https://ai-based-spam-detection.streamlit.app"
    else:
        reply = f"✅ SAFE ({(1-prob_spam)*100:.0%} legit). Good to go!"

    resp = MessagingResponse()
    resp.message(reply)
    return str(resp)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

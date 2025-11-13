from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import twilio.twiml.messaging_response as twiml

app = Flask(__name__)

# === YOUR EXISTING MODEL ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("logesh1962/sms-spam-detector")
model = AutoModelForSequenceClassification.from_pretrained("logesh1962/sms-spam-detector")
model.to(device)
model.eval()

@app.route("/whatsapp", methods=['POST'])
def whatsapp():
    # Get message from WhatsApp
    incoming_msg = request.values.get('Body', '').strip()
    
    if not incoming_msg:
        return "No message"
    
    # YOUR EXISTING PREDICTION CODE
    inputs = tokenizer(incoming_msg, return_tensors='pt', truncation=True, padding=True, max_length=96).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prob_spam = torch.softmax(outputs.logits, dim=-1)[0][1].item()
    label = "Spam/Fake" if prob_spam > 0.3 else "Legitimate"
    
    # WhatsApp Reply (160 char limit)
    if label == "Spam/Fake":
        reply = f"🚨 SPAM ({prob_spam:.0%}) Avoid links!"
    else:
        reply = f"✅ SAFE ({prob_spam:.0%})"
    
    # Send reply
    resp = twiml.MessagingResponse()
    resp.message(reply)
    return str(resp)

if __name__ == '__main__':
    app.run(debug=True)

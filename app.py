import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ------------------- MODEL LOADING -------------------
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "logesh1962/sms-spam-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ------------------- UI -------------------
st.title("Spam Detector")
st.write("Paste your message below — I'll scan for spam/scams!")

text = st.text_area(
    "Enter message:",
    height=100,
    placeholder="E.g., 'Win free iPhone! Click here...'  (type **help** for info)"
)

# ------------------- HELP COMMAND -------------------
HELP_TEXT = """
### How to use
1. **Paste any SMS / WhatsApp message** into the box.  
2. Click **Detect Spam!** – the AI will tell you **SAFE** or **SPAM** with a confidence %.

### Tips
- Short messages work best (≤ 96 tokens).  
- Links, phone numbers, or “free” giveaways raise the spam score.  
- The model is fine-tuned on ~50k English SMS.

### Commands
- `help` or `?` → show this help page.  

*Powered by RoBERTa fine-tuned on 50k SMS messages.*
"""

if text.strip().lower() in {"help", "?"}:
    st.markdown(HELP_TEXT)
    st.stop()                     # stop further execution (no spam check)

# ------------------- NORMAL SPAM CHECK -------------------
if st.button("Detect Spam!") and text.strip():
    # ----- prediction (unchanged) -----
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       padding=True, max_length=96).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prob_spam = torch.softmax(outputs.logits, dim=-1)[0][1].item()
    label = "Spam/Fake" if prob_spam > 0.3 else "Legitimate"

    # ----- display -----
    st.subheader(f"**Result: {label}**")
    st.metric("Spam Confidence", f"{prob_spam:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.success("SAFE" if label == "Legitimate" else "ALERT")
    with col2:
        st.info(f"Score: {prob_spam:.4f}")

    if label == "Spam/Fake":
        st.warning("This looks scammy — avoid clicking links or replying!")

# ------------------- FOOTER -------------------
st.sidebar.title("About")
st.sidebar.info("Powered by RoBERTa fine-tuned on 50k SMS messages.")

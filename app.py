import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ------------------- MODEL LOADING -------------------
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda_available() else 'cpu')
    model_path = "logesh1962/sms-spam-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
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
### How to Use
1. **Paste any SMS / WhatsApp / Email** into the box.  
2. Click **Detect Spam!** – AI tells you **SAFE** or **SPAM**.

### Tips
- Short messages work best.  
- Links, "free", or urgent calls = high spam.  
- Model trained on 50k+ SMS.

### Commands
- `help` or `?` → Show this help  
- `copy` → (after result) Copy verdict

*Powered by RoBERTa fine-tuned on 50k SMS.*
"""

if text.strip().lower() in {"help", "?"}:
    st.markdown(HELP_TEXT)
    st.stop()

# ------------------- SPAM DETECTION -------------------
if st.button("Detect Spam!") and text.strip():
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       padding=True, max_length=96).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prob_spam = torch.softmax(outputs.logits, dim=-1)[0][1].item()
    label = "Spam/Fake" if prob_spam > 0.3 else "Legitimate"

    # ----- Display Result -----
    st.subheader(f"**Result: {label}**")
    st.metric("Spam Confidence", f"{prob_spam:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.success("SAFE" if label == "Legitimate" else "ALERT")
    with col2:
        st.info(f"Score: {prob_spam:.4f}")

    if label == "Spam/Fake":
        st.warning("This looks scammy — avoid clicking links or replying!")

    # ------------------- COPY RESULT BUTTON -------------------
    alert_emoji = "ALERT" if label == "Spam/Fake" else "SAFE"
    safety_msg = "Avoid links & replies!" if label == "Spam/Fake" else "You're good to go!"

    result_text = f"""SPAM DETECTOR RESULT
{alert_emoji} Verdict: {label}
Confidence: {prob_spam:.1%}
{safety_msg}
Original: {text.strip()[:120]}{'...' if len(text) > 120 else ''}
App: https://spamdetectionforsms-and-mail.streamlit.app"""

    st.code(result_text, language=None)

    if st.button("Copy Result"):
        escaped_text = result_text.replace("`", "\\`")
        js = f"""
        <script>
        const text = `{escaped_text}`;
        navigator.clipboard.writeText(text).then(() => {{
            alert('Copied to clipboard!');
        }});
        </script>
        """
        st.components.v1.html(js, height=0)
        st.success("Copied! Share with friends.")

# ------------------- FOOTER -------------------
st.sidebar.title("About")
st.sidebar.info("Powered by RoBERTa fine-tuned on 50k SMS messages.")

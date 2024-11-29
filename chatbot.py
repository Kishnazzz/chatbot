import random
import time
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

# Telegram Bot Token
TELEGRAM_TOKEN = "7573682690:AAHw2NIetFbKgxbyLpSp5cne3mCur1_jLJw"
TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/"

# Load Model
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Hinglish to Hindi Transliteration
def hinglish_to_hindi(text):
    try:
        return UnicodeIndicTransliterator.transliterate(text, "hi", "hi-Latn")
    except Exception:
        return text

# Add Social Media Slang
def add_slang(response):
    slang = ["hurr", "cringe", "bakwas", "heat", "LOL", "yeh kya tha?", "no cap"]
    response += " " + random.choice(slang)
    return response

# Add Human-Like Typos
def add_typos(text):
    if random.random() < 0.3:  # 30% chance of typo
        idx = random.randint(0, len(text) - 1)
        typo = random.choice("abcdefghijklmnopqrstuvwxyz")
        return text[:idx] + typo + text[idx + 1:]
    return text

# Generate Bot Response
def generate_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return add_slang(add_typos(response))

# Telegram Bot Logic
def handle_message(update):
    chat_id = update["message"]["chat"]["id"]
    user_message = update["message"]["text"]

    # Generate Bot Reply
    user_message_hinglish = hinglish_to_hindi(user_message)
    bot_response = generate_response(user_message_hinglish)

    # Send Response Back to Telegram
    send_message(chat_id, bot_response)

def send_message(chat_id, text):
    url = TELEGRAM_URL + "sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload)

def get_updates(offset=None):
    url = TELEGRAM_URL + "getUpdates"
    params = {"offset": offset}
    response = requests.get(url, params=params)
    return response.json()

def main():
    print("Bot is running...")
    offset = None

    while True:
        updates = get_updates(offset)
        for update in updates.get("result", []):
            handle_message(update)
            offset = update["update_id"] + 1
        time.sleep(1)

if __name__ == "__main__":
    main()
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve credentials
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram_alert(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials are not configured.")
        return False

    url = f'https://api.telegram.org/bot8610023077:AAEQWn4jsI1WxglMuVar6paiHp6NPkZxi0o/sendMessage'
    payload = {
        'chat_id': CHAT_ID,
        'text': message
    }

    try:
        response = requests.post(url, data=payload, timeout=10)
        print("Status:", response.status_code)
        print("Response:", response.text)

        if response.status_code != 200:
            print("Telegram API Error:", response.text)
            return False

        return True

    except Exception as e:
        print("Telegram request failed:", e)
        return False

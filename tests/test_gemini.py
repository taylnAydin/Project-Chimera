import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # .env dosyasını oku

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")
resp = model.generate_content("Merhaba Gemini, beni duyuyor musun?")
print(resp.text)
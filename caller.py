import os
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

api_key = os.getenv("EXOTEL_API_KEY")
api_token = os.getenv("EXOTEL_API_TOKEN")
sid = os.getenv("EXOTEL_SID")
app_id = os.getenv("EXOTEL_APP_ID")

url = f"https://{api_key}:{api_token}@api.in.exotel.com/v1/Accounts/{sid}/Calls/connect"

data = {
    "From": "+919147031684",
    "To": "+919147031684",
    "CallerId": "03340100514",
    "Url": f"http://my.in.exotel.com/{sid}/exoml/start_voice/{app_id}"
}

response = requests.post(url, data=data)

print("Status:", response.status_code)
print(response.text)
print(app_id)
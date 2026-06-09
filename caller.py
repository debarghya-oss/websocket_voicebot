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
    "From": "+91xxxxxxxxx", # Your registered phone number with Exotel
    "To": "+91xxxxxxxxx", # The destination phone number you want to call
    "CallerId": "Virtual Phone Number",
    "Url": f"http://my.in.exotel.com/{sid}/exoml/start_voice/{app_id}"
}

response = requests.post(url, data=data)

print("Status:", response.status_code)
print(response.text)
print(app_id)
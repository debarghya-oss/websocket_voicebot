import requests

data = {
    'From': '+919876543210',
    'To': '+919123456789',
    'CallerId': '0XXXXXX4890',
    'Record': 'true'
}

response = requests.post(
    'https://<your_api_key>:<your_api_token>@api.exotel.com/v1/Accounts/<your_sid>/Calls/connect',
    data=data
)

print(response.json())
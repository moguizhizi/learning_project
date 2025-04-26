# Setup for predibase token access, endpoint url and headers

import os
from dotenv import load_dotenv, find_dotenv


# Initailize global variables
_ = load_dotenv(find_dotenv())

predibase_api_token = os.getenv('pb_xIy9iNM4Ul-cV5BTQv8k6A')

endpoint_url = f"{os.getenv('PREDIBASE_API_BASE', 'https://serving.app.predibase.com/d0926d85/deployments/v2/llms')}/mistral-7b"

# endpoint_url = f"{os.getenv('PREDIBASE_API_BASE', 'https://serving.app.predibase.com/6dcb0c/deployments/v2/llms')}/mistral-7b"

headers = {
    "Authorization": f"Bearer {predibase_api_token}"
}

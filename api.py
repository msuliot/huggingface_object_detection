import requests
import os
from dotenv import load_dotenv
load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN') # api key for huggingface.co in .env file
model_name = os.getenv('MODEL_NAME') # model name for huggingface.co in .env file
api_url = f"https://api-inference.huggingface.co/models/{model_name}"


def hf_api(image_path):
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    def query(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(api_url, headers=headers, data=data)
        return response.json()

    output = query(image_path)
    return output


def prettier(results):
    print("\nResults:\n")
    for item in results:
        score = round(item['score'], 3)
        label = item['label']
        location = [round(value, 2) for value in item['box'].values()]
        print(f'Detected {label} with confidence {score} at location {location}') 


def main():
    image_path = "images/10.jpg"
    return_value = hf_api(image_path)
    prettier(return_value)


if __name__ == "__main__":
    main() 
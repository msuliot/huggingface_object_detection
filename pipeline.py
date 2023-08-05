from transformers import pipeline

import os
from dotenv import load_dotenv
load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN') # api key for huggingface.co in .env file
model_name = os.getenv('MODEL_NAME') # model name for huggingface.co in .env file
pipeline_task = os.getenv('PIPELINE_TASK') # pipeline task for huggingface.co in .env file
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def hf_pipeline(image_path):
    pipe = pipeline(task=pipeline_task, model=model_name)
    output = pipe(image_path)
    return output


def prettier(results):
    for item in results:
        score = round(item['score'], 3)
        label = item['label']
        location = [round(value, 2) for value in item['box'].values()]
        print(f'Detected {label} with confidence {score} at location {location}') 


def main():
    image_path = "images/10.jpg"
    return_value = hf_pipeline(image_path)
    prettier(return_value)


if __name__ == "__main__":
    main() 
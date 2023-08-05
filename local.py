from transformers import pipeline
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

import os
from dotenv import load_dotenv
load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN') # api key for huggingface.co in .env file
model_name = os.getenv('MODEL_NAME') # model name for huggingface.co in .env file
pipeline_task = os.getenv('PIPELINE_TASK') # pipeline task for huggingface.co in .env file
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def hf_local(url):
    # This will download the model and tokenizer to your local machine and run on your local machine. 
    # saved and cached ~/.cache/huggingface
    image = Image.open(requests.get(url, stream=True).raw)

    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )


def main():
    image_url = "https://free-images.com/md/9991/kitchen_fitted_kitchen_827144.jpg"
    hf_local(image_url)


if __name__ == "__main__":
    main() 
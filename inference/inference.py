import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from peft import PeftModel
import os

# Example: Using transformers pipeline for VQA (replace with your model as needed)
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    # Load saved model, processor, and config
    model_path = os.path.abspath("vilt-finetuned-vqa")
    print(model_path)
    

    # Load model and processor, move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViltProcessor.from_pretrained(model_path)
    config = ViltConfig.from_pretrained(model_path)
    base_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", config=config, ignore_mismatched_sizes=True)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.to(device)
    model.eval()

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = processor(image, question, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                predicted_idx = logits.argmax(-1).item()
                answer = model.config.id2label[predicted_idx]
        except Exception as e:
            answer = "error"
        # Ensure answer is one word and in English (basic post-processing)
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
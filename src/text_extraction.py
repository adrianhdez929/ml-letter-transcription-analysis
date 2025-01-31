import os
import roboflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from jiwer import cer, wer
from transformers import TrOCRProcessor

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.ocr_fine_tuning import get_model, DatasetConfig, ModelConfig, get_device


# hyperparameters
CONFIDENCE = 20
OVERLAP = 45

def create_roboflow_model():
    load_dotenv()

    rf = roboflow.Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
    project = rf.workspace().project("el-makina")
    model = project.version("3").model

    return model

def extract_words_from_letter(model, letter_path):
    return model.predict(letter_path)

def order_boxes(predictions):
    boxes = []

    for prediction in predictions:
        x = int(prediction['x'])
        y = int(prediction['y'])
        width = int(prediction['width'])
        height = int(prediction['height'])
        
        # Calculate the bounding box coordinates
        x1 = x - width // 2
        y1 = y - height // 2
        x2 = x + width // 2
        y2 = y + height // 2
        
        boxes.append({
            'coordinates': (x1, y1, x2, y2),
            'center': (x, y)  # Use the center for sorting
        })

    boxes.sort(key=lambda box: (box['center'][1], -box['center'][0]))

    # Group words into lines based on their y-coordinates
    # Use a threshold to determine if words are in the same line
    y_centers = [box['center'][1] for box in boxes]
    y_centers = np.array(y_centers)
    threshold = np.mean(np.diff(np.sort(y_centers)))  # Dynamic threshold based on average vertical spacing

    # Cluster words into lines
    lines = []
    current_line = [boxes[0]]
    for box in boxes[1:]:
        if abs(box['center'][1] - current_line[-1]['center'][1]) < threshold:
            current_line.append(box)  # Same line
        else:
            lines.append(current_line)  # New line
            current_line = [box]
    if current_line:
        lines.append(current_line)

    # Sort words within each line by their x-coordinates (left to right)
    for line in lines:
        line.sort(key=lambda box: box['center'][0])

    # Flatten the sorted lines into a single list of boxes
    sorted_boxes = [box for line in lines for box in line]

    return sorted_boxes

def save_boxes(predictions, letter_path, result_path):
    original = Image.open(letter_path)
    letter_name = letter_path.split("/")[-1].split(".")[0]

    ordered = order_boxes(predictions)

    for i, box in enumerate(ordered):
        x1, y1, x2, y2 = box['coordinates']

        box = original.crop((x1, y1, x2, y2))
        box.save(f"{result_path}/{letter_name}_{i}.jpg")

def read_and_show(image_path):
    """
    :param image_path: String, path to the input image.


    Returns:
        image: PIL Image.
    """
    image = Image.open(image_path).convert('RGB')
    return image

def ocr(image, processor, model, device):
    """
    :param image: PIL Image.
    :param processor: Huggingface OCR processor.
    :param model: Huggingface OCR model.
    :param devie: Pytorch device.


    Returns:
        generated_text: the OCR'd text string.
    """
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def prepare_vectors():
    data = pd.read_csv(
        os.path.join(DatasetConfig.DATA_ROOT, 'ocr_words_train.csv'),
        header=None,
        skiprows=1,
        names=['image_filename', 'text']
    )

    return [x for x in data['image_filename'][:100]], [x for x in data['text'][:100]]

def evaluate_ocr():
    device = get_device()
    processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)
    model = get_model(device, processor)

    x_input, y_input = prepare_vectors()
    y_pred = [ocr(read_and_show(os.path.join(DatasetConfig.DATA_ROOT, x)), processor, model, device) for x in x_input]

    cer_score = cer(y_input, y_pred)
    wer_score = wer(y_input, y_pred)
    accuracy = accuracy_score(y_input, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_input, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(f"CER score: {cer_score:.4f}")
    print(f"WER score: {wer_score:.4f}")

if __name__ == "__main__":
    evaluate_ocr()

import os
import roboflow
# from inference import get_model
import numpy as np
from dotenv import load_dotenv
from PIL import Image


# hyperparameters
CONFIDENCE = 20
OVERLAP = 45

# def extract_words_from_letter(letter_path):
#     load_dotenv()

#     image = Image.open(letter_path)

#     api_key = os.getenv('ROBOFLOW_API_KEY')

#     model = get_model(model_id="el-makina/3", api_key=api_key)

#     return model.infer(image, {"confidence": CONFIDENCE, "overlap": OVERLAP})

def extract_words_from_letter(letter_path):
    load_dotenv()

    rf = roboflow.Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
    project = rf.workspace().project("el-makina")
    model = project.version("3").model

    model.confidence = CONFIDENCE
    model.overlap = OVERLAP

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

def save_boxes(predictions, letter_path):
    original = Image.open(letter_path)
    letter_name = letter_path.split("/")[-1].split(".")[0]

    ordered = order_boxes(predictions)

    for i, box in enumerate(ordered):
        x1, y1, x2, y2 = box['coordinates']

        box = original.crop((x1, y1, x2, y2))
        box.save(f"data/text_extraction/{letter_name}_{i}.jpg")
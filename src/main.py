import os
from sentiment_analysis import get_sentiment_analyzers
from text_extraction import extract_words_from_letter, ocr, save_boxes
from utils.exporters import export_images_to_csv
from utils.ocr_fine_tuning import get_device, get_model, train_ocr, get_trained_model, ModelConfig
from transformers import TrOCRProcessor

if __name__ == "__main__":
    # export_images_to_csv('data/text_extraction/pack_2')
    # train_ocr()

    input_dir = 'data/input'
    intermediate_dir = 'data/output/words'
    output_dir = 'data/output'

    processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)
    device = get_device()
    model = get_trained_model(device, processor)

    for file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, file)
        # extract words samples
        results = extract_words_from_letter(image_path)
        save_boxes(results.json()['predictions'], image_path, intermediate_dir)

        extracted = ''

        for word_box in os.listdir(intermediate_dir):
            # extract text from each box sample
            box_image_path = os.path.join(intermediate_dir, word_box)
            predicted = ocr(box_image_path, processor, model)
            extracted += f'{predicted} '

            os.remove(box_image_path)

        print(f"Text extracted from file {file}: \n {extracted}")

        # run sentiment and emotion analysis on extracted text
        sentiment, emotion = get_sentiment_analyzers()

        sentiment_result = sentiment.predict(extracted)
        emotion_result = emotion.predict(extracted)

        print(f"Setiment scores: {sentiment_result}")
        print(f"Emotion scores: {emotion_result}")

        with open(os.path.join(output_dir, file), "x") as output:
            output.write(f"Text extracted from file {file}: \n {extracted}\n")
            output.write(f"Setiment scores: {sentiment_result}\n")
            output.write(f"Emotion scores: {emotion_result}\n")

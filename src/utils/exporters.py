import os

def export_images_to_csv(path):
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            export_images_to_csv(os.path.join(path, item))
        else:
            with open('data/text_extraction/ocr_word_text.csv', 'a') as file:
                file.write(f'{os.path.join(path, item)};\n')

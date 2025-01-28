import os
import torch
import evaluate
import numpy as np
import pandas as pd
import glob as glob
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from tqdm.notebook import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

block_plot = False
plt.rcParams['figure.figsize'] = (12, 9)

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

def train_transforms(image):
    return transforms.Compose([
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
    ])(image)

@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE:    int = 48
    EPOCHS:        int = 10
    LEARNING_RATE: float = 0.00005

@dataclass(frozen=True)
class DatasetConfig:
    DATA_ROOT:     str = 'data/text_extraction/'

@dataclass(frozen=True)
class ModelConfig:
    MODEL_NAME: str = 'microsoft/trocr-small-handwritten'

class CustomOCRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

        self.df['text'] = self.df['text'].fillna('')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # The image file name.
        file_name = self.df['image_filename'][idx]
        # The text (label).
        text = self.df['text'][idx]
            
        # Read the image, apply augmentations, and get the transformed pixels.
        image = Image.open(self.root_dir + file_name).convert('RGB')
        image = train_transforms(image)
        pixel_values = self.processor(image, return_tensors='pt').pixel_values
        # Pass the text through the tokenizer and get the labels,
        # i.e. tokenized labels.
        labels = self.processor.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_target_length
        ).input_ids
        # We are using -100 as the padding token.
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize(dataset_path, df):
    all_images = df.image_filename
    all_labels = df.text
    
    plt.figure(figsize=(15, 3))
    for i in range(15):
        plt.subplot(3, 5, i+1)
        image = plt.imread(f"{dataset_path}/test_processed/images/{all_images[i]}")
        label = all_labels[i]
        plt.imshow(image)
        plt.axis('off')
        plt.title(label)
    plt.show()
    sample_df = pd.read_csv(
        os.path.join(DatasetConfig.DATA_ROOT, 'test_processed.csv'),
        header=None,
        skiprows=1,
        names=['image_filename', 'text'],
        nrows=50
    )
# visualize(DatasetConfig.DATA_ROOT, sample_df)

def compute_cer(pred, processor):
    cer_metric = evaluate.load('cer')
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
    
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    
    return {"cer": cer}

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(device, processor):
    model = VisionEncoderDecoderModel.from_pretrained(ModelConfig.MODEL_NAME)
    model.to(device)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Set special tokens used for creating the decoder_input_ids from the labels.
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # Set Correct vocab size.
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model

def get_trained_model(device, processor):
    return VisionEncoderDecoderModel.from_pretrained('trocr_handwritten/checkpoint-'+str(processor.global_step)).to(device)


def train_ocr():
    seed_everything(42)
    device = get_device()

    train_df = pd.read_csv(
        os.path.join(DatasetConfig.DATA_ROOT, 'ocr_words_train.csv'),
        header=None,
        skiprows=1,
        names=['image_filename', 'text']
    )

    test_df = pd.read_csv(
        os.path.join(DatasetConfig.DATA_ROOT, 'ocr_words_test.csv'),
        header=None,
        skiprows=1,
        names=['image_filename', 'text']
    )

    processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)
    train_dataset = CustomOCRDataset(
        root_dir=DatasetConfig.DATA_ROOT,
        df=train_df,
        processor=processor
    )
    valid_dataset = CustomOCRDataset(
        root_dir=DatasetConfig.DATA_ROOT,
        df=test_df,
        processor=processor
    )

    model = get_model(device, processor)

    optimizer = optim.AdamW(
        model.parameters(), lr=TrainingConfig.LEARNING_RATE, weight_decay=0.0005
    )

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy='epoch',
        # per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
        # per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
        fp16=True,
        output_dir='trocr_handwritten/',
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        report_to='tensorboard',
        num_train_epochs=TrainingConfig.EPOCHS,
        dataloader_num_workers=8
    )

    # Initialize trainer.
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=lambda x: compute_cer(x, processor),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator
    )

    trainer.train()
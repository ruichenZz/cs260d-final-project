import os
import pandas as pd
import torch
from transformers import AutoTokenizer

def get_dataset(args, type):
    """
    Load and preprocess the Jigsaw Unintended Bias dataset for toxicity detection.
    :param args: Arguments containing file paths, tokenizer, and preprocessing parameters.
    :param split_type: Dataset split type ('train', 'validation', 'test').
    :return: A processed DataFrame containing tokenized data and labels.
    """
    if args.dataset == 'Jigsaw':
        # Get the appropriate file path for the split

        if type == "train":
            file_path = os.path.join(args.data_dir, "train_split.csv")
        elif type == "val":
            file_path = os.path.join(args.data_dir, "val_split.csv")
        elif type == "test":
            file_path = os.path.join(args.data_dir, "test_public_expanded.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Read the dataset
        df = pd.read_csv(file_path)

        # Convert target toxicity to binary labels
        if type == "test":
            df['label'] = (df['toxicity'] >= 0.5).astype(int)  # 1 for toxic, 0 for non-toxic
        else:
            df['label'] = (df['target'] >= 0.5).astype(int)  # 1 for toxic, 0 for non-toxic

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Tokenization and preprocessing
        def tokenize_text(comment_text):
            encoded_dict = tokenizer.encode_plus(
                comment_text,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_attention_mask=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded_dict['input_ids'][0],
                'attention_mask': encoded_dict['attention_mask'][0]
            }

        # clean
        df = df.dropna()
        texts = df['comment_text'].to_list()

        # Apply tokenization to the dataset
        tokenized_data = [tokenize_text(text) for text in texts]

        # Create a new DataFrame for the tokenized data
        tokenized_df = pd.DataFrame({
            'input_ids': [data['input_ids'] for data in tokenized_data],
            'attention_mask': [data['attention_mask'] for data in tokenized_data],
            'label': df['label']
        })

        return tokenized_df

    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

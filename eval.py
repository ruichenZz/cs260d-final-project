from sklearn.metrics import classification_report
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

# Load the test set
Data_dir="./jigsaw-dataset"
model_checkpoint = "bert_model_checkpoint_on_sliced_data.pt"
test = pd.read_csv(os.path.join(Data_dir, "test_public_expanded.csv"))

# Clean the text data
test['comment_text'] = test['comment_text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n', ' ', regex=True)

# Add class labels (binary: 0 for non-toxic, 1 for toxic)
test['label'] = np.where(test['toxicity'] >= 0.5, 1, 0)

class TestDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Include labels in the output
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])  # Add labels for evaluation
        return item

# Tokenize the test set
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Use the tokenizer from the saved checkpoint
test_encodings = tokenizer(
    list(test['comment_text']),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# Create test dataset and DataLoader
test_labels = test['label'].values
test_dataset = TestDataset(test_encodings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the fine-tuned model
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained(model_checkpoint)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Evaluate the model with tqdm
model.eval()
test_predictions, test_true_labels = [], []

with torch.no_grad():
    loop = tqdm(test_loader, desc="Evaluating", leave=True)
    for batch in loop:
        # Move inputs to the device
        batch = {key: val.to(device) for key, val in batch.items()}
        
        # Extract labels from the batch
        labels = batch.pop("labels")  # Remove 'labels' from the batch for model input
        
        # Perform inference
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        # Store predictions and true labels
        test_predictions.extend(preds.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())


# Generate classification report
print(classification_report(test_true_labels, test_predictions, target_names=["Non-Toxic", "Toxic"]))

# Save predictions with test data
test['predicted_label'] = test_predictions
test.to_csv("test_predictions_crest_new.csv", index=False)
print("Test predictions saved to 'test_predictions_crest_new.csv'.")

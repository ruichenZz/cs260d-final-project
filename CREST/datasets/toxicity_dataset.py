import torch

class ToxicityIndexedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_df):
        """
        Dataset class for binary toxicity classification.
        :param tokenized_df: A Pandas DataFrame containing tokenized inputs and labels.
        """
        super().__init__()
        self.data = tokenized_df

    def __getitem__(self, index):
        """
        Get a single data point from the dataset.
        :param index: Index of the data point.
        :return: A dictionary containing input IDs, attention mask, and label.
        """
        return {
            'input_ids': self.data.iloc[index]['input_ids'],
            'attention_mask': self.data.iloc[index]['attention_mask'],
            'label': torch.tensor(self.data.iloc[index]['label'], dtype=torch.long),
            'index': index  # Include index for coreset tracking
        }

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)

    def clean(self):
        """
        Optional method to clear any cached data if necessary.
        """
        self._cachers = []

    def cache(self):
        # Cache data (e.g., save preprocessed data to disk or keep it in memory)
        print("Caching dataset...")
        self._cachers = self.data.copy()  # Example: store a copy in memory


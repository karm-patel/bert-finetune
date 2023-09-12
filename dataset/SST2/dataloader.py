import torch
from torch.utils.data import Dataset
import os
import pandas as pd

RAW_DATASET_DIR = "raw_datasets/SST_2"
class BertDataset(Dataset):
    def __init__(self, tokenizer,max_length, device=torch.device("cuda"), split="train"):
        super(BertDataset, self).__init__()
        self.data_df=pd.read_csv(os.path.join(RAW_DATASET_DIR, split + ".tsv"), delimiter='\t')
        self.tokenizer=tokenizer
        self.target=self.data_df.iloc[:,1]
        self.max_length=max_length
        self.device = device
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        
        text1 = self.data_df.iloc[index,0].lower()
        
        inputs = self.tokenizer.encode_plus(
            text1 ,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long).to(self.device),
            'mask': torch.tensor(mask, dtype=torch.long).to(self.device),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(self.device),
            'target': torch.tensor(self.data_df.iloc[index, 1], dtype=torch.long).to(self.device)
            }
    

class BertCustomDataset(Dataset):
    def __init__(self, tokenizer, max_length, sentences, device=torch.device("cuda")):
        super(BertCustomDataset, self).__init__()
        self.data_df=pd.DataFrame(sentences, columns=["text"])
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.device = device
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        
        text1 = self.data_df.iloc[index,0]
        
        inputs = self.tokenizer.encode_plus(
            text1 ,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long).to(self.device),
            'mask': torch.tensor(mask, dtype=torch.long).to(self.device),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(self.device),
            }

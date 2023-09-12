import transformers
import torch.nn as nn
import torch.nn.functional as F


class BERT(nn.Module):
    def __init__(self, d_model=768, H = 50, n_classes=2, bert_model_name="bert-base-uncased"):
        super(BERT, self).__init__()
        
        self.bert_model = transformers.BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, H),
            nn.ReLU(),
            nn.Linear(H, n_classes)
        )
        
    def forward(self,ids,mask,token_type_ids):
        # need to pass positional embeddings
        last_hidden_states, cls_processed = self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        # last_hidden_state.shape = (batch_size, sequence_length, hidden_size)
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = last_hidden_states[:, 0, :] # (Batch, MAX_length, hidden_size)
        logits = self.classifier(last_hidden_state_cls)
        return logits
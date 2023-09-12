import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torchmetrics import Accuracy
from tqdm import tqdm
from dataset.SST2.dataloader import BertDataset
from models.bert import BERT
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

ONLY_FINETUNE = 0
BATCH_SIZE = 32
MAX_SENT_LENGTH = 56
# OVERSAMPLING_RATIOS = [18, 1, 2] #[18, 1, 2]
# DATASET_NO = 0
EPOCHS = 3
# classes_dict = {0:3, 1:2, 2:5} # DATASET_NO: N_Classes
NUM_CLASSES = 2
# BERT_MODEL_NAME = "bert-large-uncased"


# lrs = [1e-4, 1e-5, 3e-5, 5e-5]
lrs = [2e-5]
models = ["bert-large-uncased"]
for lr,BERT_MODEL_NAME in itertools.product(lrs,models):
    print(f"============ {BERT_MODEL_NAME} LR-{lr} ===============")
    D_MODEL = 768 if "base" in BERT_MODEL_NAME else 1024
    H = 50 if "base" in BERT_MODEL_NAME else 100
    tokenizer = transformers.BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # train dataset
    train_dataset= BertDataset(tokenizer, max_length=MAX_SENT_LENGTH, split="train")
    train_dataloader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE)
    print(len(train_dataset), len(train_dataloader))

    # dev dataset
    dev_dataset= BertDataset(tokenizer, max_length=MAX_SENT_LENGTH, split="dev")
    dev_dataloader=DataLoader(dataset=dev_dataset,batch_size=BATCH_SIZE)
    print(len(dev_dataset), len(dev_dataloader))

    # test dataset
    test_dataset= BertDataset(tokenizer, max_length=MAX_SENT_LENGTH, split="test")
    test_dataloader=DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE)
    print(len(test_dataset), len(test_dataloader))

    # testing data loaders
    next(iter(train_dataloader))

    # model
    model=BERT(n_classes=NUM_CLASSES, bert_model_name=BERT_MODEL_NAME, d_model=D_MODEL, H=H).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer= optim.Adam(model.parameters(),lr=lr)

    # only finetune classification head
    if ONLY_FINETUNE:
        for param in model.bert_model.parameters():
            param.requires_grad = False
    # training
    train_losses = []
    dev_losses = []
    best_valid_accs, best_macro_f1 = 0, 0
    for epoch in range(EPOCHS):
        loop=tqdm(enumerate(train_dataloader),leave=False,total=len(train_dataloader))
        model.train()
        print(epoch)
        total_matches = 0
        for batch, dl in loop:
            ids, token_type_ids, mask, label = dl['ids'], dl['token_type_ids'], dl['mask'], dl['target']

            optimizer.zero_grad()
            output = F.log_softmax(model(ids=ids,mask=mask,token_type_ids=token_type_ids), dim=1) # (B, 2)


            loss=loss_fn(output,label)
            loss.backward()
            train_losses.append(loss)

            optimizer.step()
            pred = torch.argmax(output, dim=-1) # (B,1)
            accuracy = Accuracy(task = "multiclass", num_classes=NUM_CLASSES).to(device)(pred, label)
            total_matches += (torch.sum(pred == label)).item()

            # Show progress while training
            loop.set_description_str(f"Train - Epoch={epoch}/{EPOCHS} loss={loss.item()} acc={accuracy}")

        print(f"Train Accuracy :{epoch} = {total_matches/len(train_dataset)}") 

        # dev dataset
        model.eval()
        loop=tqdm(enumerate(dev_dataloader),leave=False,total=len(dev_dataloader))
        total_matches = 0
        final_pred = []
        final_label = []
        with torch.no_grad():
            for batch, dl in loop:
                ids, token_type_ids, mask, label = dl['ids'], dl['token_type_ids'], dl['mask'], dl['target']
                final_label.append(label)
                # output=model(ids=ids,mask=mask,token_type_ids=token_type_ids)
                output = F.log_softmax(model(ids=ids,mask=mask,token_type_ids=token_type_ids), dim=1)
        #         label = label.type_as(output)

                loss=loss_fn(output,label)
                dev_losses.append(loss)

                pred = torch.argmax(output, dim=-1)
                final_pred.append(pred)

                matches = torch.sum(pred == label)
                total_matches += matches.item()

                # Show progress while training
                loop.set_description(f'VAL - Epoch={epoch}/{EPOCHS} loss={loss.item()}')

            val_acc = total_matches/len(dev_dataset)
            
            final_pred = torch.concat(final_pred)
            final_label = torch.concat(final_label)
            macro_f1 = f1_score(final_label.to(cpu_device), final_pred.to(cpu_device), average='macro')

            print(f"Validation Accuracy :{epoch} = {val_acc}")
            print(f"Macro F1 :{epoch} = {macro_f1}")

            if val_acc > best_valid_accs:
                best_valid_accs = val_acc
                best_macro_f1 = macro_f1
                best_epoch = epoch
                cf = confusion_matrix(final_label.to(cpu_device), final_pred.to(cpu_device))

    metrics = {
        "train_losses": train_losses,
        "val_losses": dev_losses
    }

    model_name = "BERT"
    EMB_MODEL = "BERT"
    # save the plots
    # for d in range(3):
    #     if not os.path.exists("plots/D{d}"):
    #         os.makedirs("plots/D{d}")

    plt.figure()
    plt.plot(torch.tensor(metrics["train_losses"]).to("cpu"), label="Train loss")
    plt.title(f"Epochs - {EPOCHS} Batch Size - {BATCH_SIZE} Lr - {lr}")
    plt.legend()
    plt.savefig(f"plots/{model_name}_train_losses_{EPOCHS}_{lr}_SST2.pdf")

    plt.figure()
    plt.plot(torch.tensor(metrics["val_losses"]).to("cpu"), label="Val loss")
    plt.title(f"Epochs - {EPOCHS} Batch Size - {BATCH_SIZE} Le -{lr}")
    plt.legend()
    plt.savefig(f"plots/{model_name}_val_losses_{EPOCHS}_{lr}_SST2.pdf")

    print(cf)

    # save the results
    if not os.path.exists("results/"):
        os.makedirs("results/")

    with open(f"results/{BERT_MODEL_NAME}_SST2.txt", "a") as fp:
        fp.write(f"BERT MODEL NAME = {BERT_MODEL_NAME}\n")
        fp.write(f"ONLY_FINETUNE = {ONLY_FINETUNE}\n")
        fp.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
        # fp.write(f"OVERSAMPLING_RATIO = {OVERSAMPLING_RATIOS}\n")
        fp.write(f"Learning Rate = {lr}\n")
        fp.write(f"EPOCHS = {EPOCHS}\n")
        fp.write(f'BEST VAL ACCURACY - {best_valid_accs}\n')
        fp.write(f'BEST MACRO F1 - {best_macro_f1}\n')
        fp.write(f'BEST EPOCH - {best_epoch}\n')
        fp.write(f'{cf}\n')
        fp.write("====================================================================\n")

    ckpt_name = f"{BERT_MODEL_NAME}_SST2_{EPOCHS}_{lr}_FULL"
    torch.save(model.state_dict(), f"ckpts/{ckpt_name}.pt")
    print([model.state_dict()[key].shape for key in model.state_dict()])
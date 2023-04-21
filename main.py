import platform
import torch
import pandas as pd
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Load the Quora Question Pairs dataset
# https://www.kaggle.com/c/quora-question-pairs/data

df = pd.read_csv("train.csv", keep_default_na=False)

# Split the dataset into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


if torch.cuda.is_available():
    device_type = "cuda"
elif torch.has_mps:
    device_type = "mps"
else:
    device_type = "cpu"


device = torch.device(device_type)

q1s = train_df["question1"].tolist()

for i, q1 in enumerate(q1s):
    if type(q1) != str:
        print(train_df["id"].values[i])
        print(i, q1)

# Load the tokenizer
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

# Tokenize the questions and convert them to tensors
train_encodings = tokenizer(list(train_df['question1'].values), list(train_df['question2'].values), truncation=True, padding=True)
val_encodings = tokenizer(list(val_df['question1'].values), list(val_df['question2'].values), truncation=True, padding=True)

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings["input_ids"]), 
    torch.tensor(train_encodings["attention_mask"]), 
    torch.tensor(list(train_df["is_duplicate"]))
    )

val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(val_encodings["input_ids"]),
    torch.tensor(val_encodings["attention_mask"]),
    torch.tensor(list(val_df["is_duplicate"]))
    )


# Load the ALBERT model
model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)
model.to(device)

# Define the optimizer and the learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)


def train_loop(dataloader, optimizer, epoch):
    model.train()
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())

def eval_loop(dataloader, epoch):
    model.eval()
    total_correct = 0
    total_samples = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                predictions = torch.argmax(outputs.logits, dim=1)
                total_correct += torch.sum(predictions == labels)
                total_samples += len(labels)
            
            tepoch.set_postfix(accuracy=total_correct/total_samples)
    print(f"Epoch {epoch} - Accuracy: {total_correct/total_samples}")
        
    
    # Return the accuracy
    return total_correct / total_samples 


# Train the model
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

accuracy = eval_loop(val_dataloader, 0)

for epoch in range(3):
    train_loop(train_dataloader, optimizer, epoch+1)
    scheduler.step()
    accuracy = eval_loop(val_dataloader, epoch+1)
    print(f"Epoch {epoch+1} - Accuracy: {accuracy}")

# Save the model
model.save_pretrained("albert-pretrained-v1")
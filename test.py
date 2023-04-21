import torch
import pandas as pd
from transformers import AutoTokenizer, AlbertModel
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Load the Quora Question Pairs dataset
# https://www.kaggle.com/c/quora-question-pairs/data

df = pd.read_csv("train.csv", keep_default_na=False).head(3000)

if torch.cuda.is_available():
    device_type = "cuda"
elif torch.has_mps:
    device_type = "mps"
else:
    device_type = "cpu"


device = torch.device(device_type)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

# Tokenize the questions and convert them to tensors
inputs = []
inputs.append(tokenizer(list(df['question1'].values), truncation=True, padding=True, return_tensors='pt').to(device))
inputs.append(tokenizer(list(df['question2'].values), truncation=True, padding=True, return_tensors='pt').to(device))

# print(list(map(tokenizer.convert_ids_to_tokens, inputs[0]["input_ids"].to('cpu'))), list(map(tokenizer.convert_ids_to_tokens, inputs[1]["input_ids"].to('cpu'))))

model = AlbertModel.from_pretrained("albert-base-v2").to(device)

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
with torch.no_grad():
    q1embedding = model(**inputs[0])[0]
    q2embedding = model(**inputs[1])[0]
    output = cos(q1embedding[:, 0, :], q2embedding[:, 0, :])
    print(q1embedding[:, 0, :], q2embedding[:, 0, :])

output = output.to("cpu")
cutoffs = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
scores = np.empty((len(cutoffs), 3))
for i, cutoff in enumerate(cutoffs):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for j in range(len(df)):
        if output[j] >= cutoff:
            if df["is_duplicate"].iloc[j] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if df["is_duplicate"].iloc[j] == 1:
                fn += 1
            else:
                tn += 1
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f1 = 2 * (p * r) / (p + r)
    scores[i] = [p, r, f1]

t = np.arange(len(df))
plt.scatter(t, output.to("cpu"), c=list(df["is_duplicate"]), cmap=ListedColormap(["green","red"]))
plt.savefig('test.png')

plt.clf()
plt.plot(cutoffs, scores[:, 0], label="Precision")
plt.plot(cutoffs, scores[:, 1], label="Recall")
plt.plot(cutoffs, scores[:, 2], label="F1")
plt.legend()
plt.xlabel("Cosine Similarity Cutoff")
plt.ylabel("Score")

plt.savefig("test2.png")

# Dependencies importation

import logging
#import os
#import urllib.request
import random
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from sentence_transformers.readers import InputExample
from sentence_transformers import LoggingHandler, losses, SentenceTransformer
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
#from datetime import datetime

def get_dataframe_from_ng() : 
    df_list = []
    for subset in ["train", "test"] : 
        data = fetch_20newsgroups(subset=subset, remove=("headers", "footers","quotes"), shuffle=True)
        df = pd.DataFrame({
            "data" : data.data,
            "labels" : data.target
        })
        df['data'].replace('', np.nan, inplace=True)
        df.dropna(inplace=True)
        df_list.append(df)
    return df_list

# We split data in train eval and test dataframe 
df_train, df_test = get_dataframe_from_ng()
df_test, df_eval = train_test_split(
    df_test,
    test_size=0.5,
    shuffle = True,
)

print(
    f" train size : {len(df_train)}" \
    f" eval size : {len(df_eval)}" \
    f" test size : {len(df_test)}" \
)

# Plot occurence of each class in the train data
count = Counter(df_train.labels)
bar = [str(x) for x in count.keys()]
height = list(count.values())
plt.bar(bar,count.values())
plt.xlabel("20_NewsGroups class")
plt.ylabel("Occurence")
plt.show()

## Number of tokens by sentence

tokens = [len(sentence.split()) for sentence in df_train.data]
pd.Series(tokens).describe()


####################################################################
## Fine-tuning using ```distilbert-base-nli-mean-tokens```

# In the last version of ```sentence_transformers```, according to this 
# <a href  = https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark.py >link</a>, 
# We won't need a Dataset class to train with ```sentence_transformers```. Juste a list of ```InputExample``` is enougth. 
# Let's create it

####################################################################

def get_input_example(df : pd.DataFrame) : 

    examples = []
    for i, (text, label) in enumerate(zip(df.data.values, df.labels.values)) :
        examples.append(InputExample(guid=i, texts=[text], label=label))

    return examples


def get_triplets_input_example(df : pd.DataFrame) : 

    index = df.index.values
    
    triplets_input_examples = []

    for ind, anchor, anchor_label in zip(df.index, df.data, df.labels) : 

        positive_list = index[index!=ind][df["labels"][index!=ind]==anchor_label]
        positive_item = random.choice(positive_list)
        positive_example = df["data"].loc[positive_item]

        negative_list = index[index!=ind][df["labels"][index!=ind]!=anchor_label]
        negative_item = random.choice(negative_list)
        negative_example = df["data"].loc[negative_item]

        triplets_input_examples.append(InputExample(texts = [anchor, positive_example, negative_example], label = anchor_label))

    return triplets_input_examples

# Logging Configuration 
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Continue training distilbert-base-nli-mean-tokens on 20news_groups data
MODEL_NAME = 'distilbert-base-nli-mean-tokens'

### Create a torch.DataLoader that passes training batch to our model
BATCH_SIZE = 16

# Load pretrained model
model = SentenceTransformer(MODEL_NAME)

logging.info("Read 20Newsgroups data")

# Get training_examples
train_examples = get_triplets_input_example(df_train)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE )
train_loss = losses.TripletLoss(model)


### Accuracy before model fine tuning
eval_triplets_examples = get_triplets_input_example(df_eval)

logging.info("Evaluation ...")
dev_evaluator = TripletEvaluator.from_input_examples(eval_triplets_examples,  name='20_ng_eval')
logging.info("Accuracy :")
dev_evaluator(model)


num_epochs = 1
output_path = "./models"

### Model Fune tuning
warmup_steps = int(len(train_examples) * num_epochs / BATCH_SIZE * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=output_path,
)


### Evaluating model performance before model fine tuning
test_triplets_examples = get_triplets_input_example(df_test)

logging.info("Evaluation on test")
test_evaluator = TripletEvaluator.from_input_examples(test_triplets_examples,  name='20_ng_test')
logging.info("Accuracy : ")
test_evaluator(model)

# Load sBert pretrained model
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(SBERT_MODEL_NAME)

logging.info("Evaluation on test")
dev_evaluator = TripletEvaluator.from_input_examples(eval_triplets_examples,  name='20_ng_test')
logging.info("Accuracy : ")
dev_evaluator(model)
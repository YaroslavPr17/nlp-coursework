import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.nlp.classification import Pipeline


class ReviewDataset(Dataset):
    def __init__(self, reviews, tokenizer, labels=False, max_model_input_length=512):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_model_input_length = max_model_input_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews.iloc[idx]
        if not isinstance(self.labels, bool):
            label = self.labels.iloc[idx]
        review_tokenized = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_model_input_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        input_ids = review_tokenized['input_ids'].flatten()
        attn_mask = review_tokenized['attention_mask'].flatten()
        
        ret = {
            'review': review,
            'input_ids': input_ids,
            'attention_mask': attn_mask
        }         
        if not isinstance(self.labels, bool):
            ret['label'] = label 
        return ret


class BertClassifier:
    def __init__(self, checkpoint, n_classes=2):

        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=False,
                                                                        ignore_mismatched_sizes=True)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = self.model.config.max_position_embeddings
        self.out_features = self.model.config.pooler_fc_size
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)

        self.all_losses = []
        self.epoch_losses = []
        self.epoch_acc = []

    def fit(self, train_dataloader: torch.utils.data.DataLoader):
        self.model.train()
        losses = []
        correct_predictions = 0

        t = tqdm(train_dataloader, file=sys.stdout, ncols=100)

        for data in t:
            input_ids = data['input_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device).to(float)
            labels = data['label'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = outputs.logits.argmax(dim=1)

            loss = self.loss_fn(outputs.logits, labels)

            correct_predictions += torch.sum(preds == labels)

            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            t.set_postfix(ordered_dict={'loss': loss.item()}, refresh=True)

        train_acc = correct_predictions.double() / len(train_dataloader.dataset)
        train_loss = np.mean(losses)
        self.all_losses.extend(losses)
        self.epoch_losses.append(train_loss)
        self.epoch_acc.append(train_acc)
        return train_acc, train_loss

    def evaluate(self, test_dataloader: DataLoader):
        self.model.eval()
        losses = []
        correct_predictions = 0

        all_preds = []
        all_labels = []

        t = tqdm(test_dataloader, file=sys.stdout, ncols=100)

        with torch.no_grad():
            for data in t:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, labels)
                correct_predictions += torch.sum(preds == labels)

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

                losses.append(loss.item())

                t.set_postfix(ordered_dict={'loss': loss.item()}, refresh=True)

        print('Classification report:')
        print(classification_report(all_labels, all_preds))

        val_acc = correct_predictions.double() / len(test_dataloader.dataset)
        val_loss = np.mean(losses)
        return val_acc.item(), val_loss

    def train(self, n_epochs: int,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              pretrain_test: bool =False):
        try:
            if pretrain_test:
                print('Pre-training test:')
                val_acc, val_loss = self.evaluate(test_dataloader)
                print(f'Test loss {val_loss} accuracy {val_acc}')
                print('-' * 10)

            for epoch in range(n_epochs):
                print(f'Epoch {epoch + 1}/{n_epochs}')
                train_acc, train_loss = self.fit(train_dataloader)
                print(f'Train loss {train_loss} accuracy {train_acc}')

                val_acc, val_loss = self.evaluate(test_dataloader)
                print(f'Test loss {val_loss} accuracy {val_acc}')
                print('-' * 10)

                # self.scheduler.step()

        except KeyboardInterrupt:
            print('Training was manually stopped. ')
            
    
class BertEmbedder:
    def __init__(self, model, tokenizer, device=None):

        self.model = model
        self.tokenizer = tokenizer

        self.max_len = self.model.config.max_position_embeddings
        self.out_features = self.model.config.pooler_fc_size
        self.model.dropout = torch.nn.Sequential()
        self.model.classifier = torch.nn.Sequential()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if not device else torch.device(device) 
        self.model.to(self.device)
        

    def build(self, train_dataloader: torch.utils.data.DataLoader):
        self.model.eval()

        all_embeddings = np.array([])

        t = tqdm(train_dataloader, file=sys.stdout)

        for data in t:
            with torch.no_grad():
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device).to(float)
                
                embeddings = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits

                all_embeddings = np.append(all_embeddings, embeddings.cpu().numpy())

        all_embeddings = all_embeddings.reshape(-1, self.out_features)
        
        return all_embeddings



class BertLogRegClassifier:
    def __init__(self, checkpoint):

        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.max_len = self.model.config.max_position_embeddings
        self.out_features = self.model.config.pooler_fc_size
        self.model.dropout = torch.nn.Sequential()
        self.model.classifier = torch.nn.Sequential()

        self.classifier = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.all_train_embeddings = None
        self.all_test_embeddings = None

    def fit(self, train_dataloader: torch.utils.data.DataLoader):
        self.model.eval()

        self.all_train_embeddings = np.array([])
        all_labels = []

        t = tqdm(train_dataloader, file=sys.stdout, ncols=100)

        for data in t:
            with torch.no_grad():
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device).to(float)
                labels = data['label'].to(self.device)

                embeddings = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits

                self.all_train_embeddings = np.append(self.all_train_embeddings, embeddings.cpu().numpy())
                all_labels.extend(labels.tolist())

        self.all_train_embeddings = self.all_train_embeddings.reshape(-1, self.out_features)

        self.classifier.fit(self.all_train_embeddings, all_labels)

    def predict(self, test_input, print_report=False):
        self.model.eval()

        self.all_test_embeddings = np.array([])
        all_labels = []

        t = tqdm(test_input, file=sys.stdout, ncols=100)

        for data in t:
            with torch.no_grad():
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device).to(float)
                labels = data['label']

                embeddings = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits

                self.all_test_embeddings = np.append(self.all_test_embeddings, embeddings.cpu().numpy())

                all_labels.extend(labels.tolist())

        self.all_test_embeddings = self.all_test_embeddings.reshape(-1, self.out_features)

        all_preds = self.classifier.predict(self.all_test_embeddings)

        if print_report:
            print(classification_report(all_labels, all_preds))

        return all_preds


def get_df_by_person(data: pd.DataFrame, name: str) -> pd.DataFrame:
    filtered = data[data['ne'].str.contains(name, case=False)].sort_values(by='n_sents', ascending=False)
    return filtered


def get_df_by_film_and_person(data: pd.DataFrame, film_id: int, name: str) -> pd.DataFrame:
    df_with_person = get_df_by_person(data, name)
    filtered = df_with_person[df_with_person['film_id'] == film_id].sort_values(by='n_sents', ascending=False)
    return filtered


def collect_sents_to_summarize(data: pd.DataFrame, n_sents: int = 100) -> List[str]:
    all_sents = []
    for sents in data['occurrences']:
        all_sents.extend(sents)

    if not all_sents:
        print()
        return all_sents
    all_sents = np.array(all_sents)
    pl = Pipeline('classification', model_type='linear', preprocess=True)
    _, dct = pl(pd.Series(all_sents), return_confs=True)
    confs = dct['confs']
    to_summarize = []
    to_summarize.extend(all_sents[confs < 0].tolist())
    _n = n_sents - len(to_summarize)
    _n = _n if _n < len(confs) else len(confs) - len(to_summarize)
    to_summarize.extend(all_sents[np.argpartition(confs, -_n)[-_n:]].tolist())

    return to_summarize


def split_opinions_to_chunks(tokenizer,
                             data: pd.DataFrame = None,
                             sentences: List[str] = None,
                             model_max_input: int = 600,
                             show_info: bool = False):
    if sentences is not None:
        all_sents = sentences
    elif data is not None:
        all_sents = list(data['occurrences'].sum())
    else:
        raise ValueError('No data is provided!')

    chunks = ['']
    chunks_length = [0]

    for sent in all_sents:
        _new_length = chunks_length[-1] + len(tokenizer.encode(sent))
        if _new_length > model_max_input:
            if len(tokenizer.encode(sent)) > model_max_input:
                print('The sentence has length greater that max model input, thus will be skipped', file=sys.stderr)
                continue
            chunks.append('')
            chunks_length.append(0)
            _new_length = chunks_length[-1] + len(tokenizer.encode(sent))

        chunks_length[-1] = _new_length
        chunks[-1] += ' ' + sent

    if show_info:
        import matplotlib.pyplot as plt

        print('Overall number of sentences:', len(all_sents))
        print('Number of chunks:', len(chunks))

        plt.figure(figsize=(10, 3))

        plt.plot(chunks_length)
        plt.title('The number of tokens across all chunks')
        plt.xlabel('Chunk number')
        plt.ylabel('Number of tokens')

        plt.show()

    return chunks

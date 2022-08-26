import streamlit as st
import numpy as np
import pandas as pd

# modeling
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    )
from transformers.optimization import Adafactor

# aesthetics
from IPython.display import Markdown, display, clear_output
import re
import warnings
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
seed_everything(25429)

# scoring
import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# loading the model
hug = 't5-small'
t5tokenizer = T5Tokenizer.from_pretrained(hug)
t5model = T5ForConditionalGeneration.from_pretrained(hug, return_dict=True)

# defining tokens
SEP_TOKEN = '<sep>'
MASK_TOKEN = '[MASK]'
MASKING_CHANCE = 0.1

class DataEncodings(Dataset):
    '''
    tokenizes, pads, and adds special tokens
    '''
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        source_max_token_len: int,
        target_max_token_len: int
        ):
        self.tokenizer = t5tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int):
        data_row = self.data.iloc[index]
        # adds a random mask for answer-agnostic qg
        if np.random.rand() > MASKING_CHANCE:
            answer = data_row['answer']
        else:
            answer = MASK_TOKEN
            
        source_encoding = t5tokenizer(
            f"{answer} {SEP_TOKEN} {data_row['context']}",
            max_length= self.source_max_token_len,
            padding='max_length',
            truncation= True,
            return_attention_mask=True,
            return_tensors='pt'
            )
    
        target_encoding = t5tokenizer(
            f"{data_row['answer']} {SEP_TOKEN} {data_row['question']}",
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors='pt'
            )

        labels = target_encoding['input_ids']  
        labels[labels == 0] = -100 # masked

        encodings = dict(
            answer = data_row['answer'],
            context = data_row['context'],
            question = data_row['question'],
            input_ids = source_encoding['input_ids'].flatten(),
            attention_mask = source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
        )
        
        return encodings

class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        tokenizer,
        batch_size,
        source_max_token_len: int,
        target_max_token_len: int
        ): 
        super().__init__()
        self.batch_size = batch_size
        self.train = train
        self.val = val
        self.tokenizer = t5tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self):
        self.train_dataset = DataEncodings(self.train, self.tokenizer, self.source_max_token_len, self.target_max_token_len)
        self.val_dataset = DataEncodings(self.val, self.tokenizer, self.source_max_token_len, self.target_max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size=batch_size, num_workers=0)

# hyperparameters
num_epochs = 16
batch_size = 32
learning_rate = 0.001

# model 
class T5Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = t5model
        self.model.resize_token_embeddings(len(t5tokenizer)) # resizing after adding new tokens to the tokenizer

    # feed forward pass
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    # train model and compute loss
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    # gets model predictions, returns loss
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True, batch_size=batch_size)
        return {'val loss': loss}
    
    # def validation_epoch_end(self, outputs):
    #     # outputs = list of dictionaries to print loss
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'avg_val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return Adafactor(model.parameters(), scale_parameter=False, relative_step=False, lr=learning_rate) 

def generate(model: T5Model, answer:str, context:str) -> str:
    source_encoding = t5tokenizer(
        f"{answer} {SEP_TOKEN} {context}",
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids=model.model.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_beams=20,
        max_length=126,
        repetition_penalty=2.5,
        length_penalty=0.8,
        temperature=0.6,
        early_stopping=True,
        use_cache=True
    )

    preds = {
        t5tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }

    return ''.join(preds)

def show_result(generated:str, answer:str, context:str, original_question:str=''):

    regex = r"(?<=>)(.*?)(?=<)"
    matches = re.findall(regex, generated)
    matches[1] = matches[1][5:]
    final = {cat: match.strip() for cat, match in zip(['Answer', 'Question'], matches)}
    st.title('Generated')
    st.write(final)
    # if original_question:
    #     printBold('Original Question')
    #     print(original_question)
    #     gen = nlp(matches[1])
    #     ori = nlp(original_question)
    #     bleu_score = sentence_bleu(matches[1], original_question, smoothing_function=SmoothingFunction().method5)
    #     cs_score = ori.similarity(gen)
    #     printBold('Scores')
    #     print(f"BLEU: {bleu_score}")
    #     print(f'Cosine Similarity: {cs_score}')
    #     return bleu_score, cs_score

# streamlit app
st.title('Question Generation From Text')

with st.spinner('Loading Model...'):
    model = T5Model
    best_model_dir = '-GA-Stuff/DSI-working-folder/QG-System/checkpoints/t5-chkpt-v2.ckpt'
    best_model = model.load_from_checkpoint(best_model_dir)
    # best_model = model.load_from_checkpoint(callback.best_model_path)
    best_model.freeze()

with st.form('my_form'):
    context = st.text_input('Enter a context passage for question generation:', 'The capital of France is Paris.')
    answer = st.text_input('Give a correct answer, or [MASK] for unsupervised generation:', 'Paris')
    # question = st.text_input('Question', 'What is the capital of France?')
    # original_question = st.text_input('Original Question', 'What is the capital of France?')
    submitted = st.form_submit_button('Generate')

with st.spinner('Generating...'):
    if submitted:
        generated = generate(best_model, answer, context)
        show_result(generated, answer, context)
        
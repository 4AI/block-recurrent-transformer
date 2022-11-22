import re
import sys

# from datasets import load_dataset
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, DataLoader
from torch import nn
from torch.nn.functional import one_hot, cross_entropy
from torch.optim import Adam
from torchinfo import summary
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer
from x_transformers.x_transformers import TokenEmbedding
import wandb

from block_recurrent_transformer import BlockRecurrentAttention, long_sequence_splitter



def load_wikidata(fpath, min_len=10, max_length=None):
    with open(fpath) as reader:
        docs = re.split(r'\=[^=]+\=', reader.read())
        docs = [doc.strip() for doc in docs if doc.strip() and len(doc.split()) > min_len]
        if max_length is not None:
            docs = [' '.join(doc.split()[:max_length]) for doc in docs]
    return docs


class WikiDataset:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i: int):
        return self.data[i]

    def __len__(self):
        return len(self.data)


class BlockRecurrentDecoder(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, num_tokens, dim):
        super().__init__()
        self.embed = TokenEmbedding(dim, num_tokens)
        self.attn = BlockRecurrentAttention(dim, dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, state=None):
        x, state = self.attn(self.embed(x), state)
        x = self.to_logits(self.norm(x))
        return x, state



def setup_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # print('>>> tokenizer size:', len(tokenizer))
    return tokenizer



def train_eval( train_data, test_data, tokenizer, config, epochs=20, verbose_interval=100, block_dim=1100):
    model = BlockRecurrentDecoder(len(tokenizer), block_dim)
    model.to(device)
    opt = Adam(model.parameters())
    
    # model summary
    summary(model)
    
    train_data = WikiDataset(train_data)
    test_data = WikiDataset(test_data)
    best_ppl = float('inf')
    for epoch in range(epochs):
        # train
        model.train()
        i = 0
        all_train_ppls = []
        data_loader = DataLoader(train_data,  batch_size = config.batch_size, sampler=RandomSampler(train_data), pin_memory=True)
        total_steps = len(data_loader)
        print('total train steps:', total_steps)
        for raw_batch in data_loader:
            state = None
            article_batch = tokenizer(raw_batch, return_tensors='pt', max_length=256, truncation='longest_first', padding=True)['input_ids']
            for text in long_sequence_splitter(article_batch, config.window_len):
                inputs = text[:, :-1].to(device)
                targets = text[:, 1:].to(device)
                preds, state = model(inputs, state)
                preds = preds.permute((0, 2, 1))
                # to avoid runtime error: https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
                state = state.detach()
                opt.zero_grad()
                loss = cross_entropy(preds, targets)

                ppl = torch.exp(loss)
                loss_score = loss.detach().cpu().item()
                ppl = ppl.detach().cpu().item()
                all_train_ppls.append(ppl)
                if i % verbose_interval == 0:
                    print(f'Epoch: {epoch}, steps: {round(i/total_steps) * 100}% ({i}/{total_steps}), loss: {loss_score}, ppl: {ppl}')

                loss.backward()
                opt.step()
                i += 1
        print(f'(train) Epoch {epoch}, avg train ppl: {sum(all_train_ppls)/len(all_train_ppls)}')

        # eval
        model.eval()
        with torch.no_grad():
            data_loader = DataLoader(test_data,  batch_size = config.batch_size, shuffle=False,  pin_memory=True)
            total_steps = len(data_loader)
            print('total test steps:', total_steps)
            all_ppls = []
            i = 0
            for raw_batch in data_loader:
                state = None
                article_batch = tokenizer(raw_batch, return_tensors='pt', max_length=config.seq_len, truncation='longest_first', padding=True)['input_ids']
                for text in long_sequence_splitter(article_batch, config.window_len):
                    inputs = text[:, :-1].to(device)
                    targets = text[:, 1:].to(device)
                    preds, state = model(inputs, state)
                    # to avoid runtime error: https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
                    state = state.detach()
                    preds = preds.permute((0, 2, 1))
                    loss = cross_entropy(preds, targets)

                    ppl = torch.exp(loss)
                    loss = loss.detach().item()
                    ppl = ppl.detach().item()
                    all_ppls.append(ppl)
                    if i % verbose_interval == 0:
                        print(f'(eval) Epoch: {epoch}, steps: {round(i/total_steps) * 100}% ({i}/{total_steps}), loss: {loss}, ppl: {ppl}')

                    i += 1
            avg_test_ppl = sum(all_ppls) / len(all_ppls)
            if best_ppl > avg_test_ppl:
                best_ppl = avg_test_ppl
            print(f'Epoch {epoch}, avg test ppl: {avg_test_ppl}, best test ppl: {best_ppl}')


if __name__ == '__main__':
    device = 'cuda:0'
    tokenizer = setup_tokenizer()
    config = OmegaConf.load('configs/base.yaml')
    train_data = load_wikidata('data/wikitext-103/train.txt', max_length=config.seq_len)
    test_data = load_wikidata('data/wikitext-103/test.txt', max_length=config.seq_len)
    train_eval(train_data, test_data, tokenizer, config)

import re
import sys

from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, DataLoader
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer
from x_transformers.x_transformers import TokenEmbedding
import wandb

from block_recurrent_transformer import BlockRecurrentAttention, long_sequence_splitter



def load_wikidata(fpath, min_len=10):
    with open(fpath) as reader:
        docs = re.split(r'\=[^=]+\=', reader.read())
        docs = [doc.strip() for doc in docs if doc.strip() and len(doc.split()) > min_len]
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
    return tokenizer



def train_eval( train_data, test_data, tokenizer, config, epochs=5, verbose_interval=100):
    model = BlockRecurrentDecoder(len(tokenizer), 512)
    model.to(device)
    opt = Adam(model.parameters())
    train_data = WikiDataset(train_data)
    test_data = WikiDataset(test_data)
    for epoch in range(epochs):
        # train
        model.train()
        i = 0
        data_loader = DataLoader(train_data,  batch_size = config.batch_size, sampler=RandomSampler(train_data), pin_memory=True)
        for raw_batch in tqdm(data_loader):
            state = None
            article_batch = tokenizer(raw_batch, return_tensors='pt', padding=True)['input_ids']
            for text in tqdm(long_sequence_splitter(article_batch, config.window_len)):
                inputs = text[:, :-1]
                targets = text[:, 1:]
                preds, state = model(inputs, state)
                loss = cross_entropy(preds, targets)
                loss.backward()
                opt.step()
                '''
                preds, state = preds.detach(), state.detach()
                preds.to('cpu')
                '''
                ppl = torch.exp(loss)
                loss = loss.detach().items()
                ppl = ppl.detach().items()
                sys.stdout.write(f'Epoch {epoch}, loss: {loss}, ppl: {ppl}')
                i += 1
    
        # eval
        print('start to evaluate...')
        model.eval()
        with torch.no_grad():
            data_loader = DataLoader(test_data,  batch_size = config.batch_size, shuffle=False,  pin_memory=True)
            all_ppls = []
            i = 0
            for raw_batch in tqdm(data_loader):
                state = None
                article_batch = tokenizer(raw_batch, return_tensors='pt', padding=True)['input_ids']
                for text in tqdm(long_sequence_splitter(article_batch, config.window_len)):
                    inputs = text[:, :-1]
                    targets = text[:, 1:]
                    preds, state = model(inputs, state)
                    loss = cross_entropy(preds, targets)

                    ppl = torch.exp(loss)
                    loss = loss.detach().items()
                    ppl = ppl.detach().items()
                    all_ppls.append(ppl)
                    sys.stdout.write(f'Epoch {epoch}, loss: {loss}, ppl: {ppl}')
                    i += 1
            print(f'Epoch {epoch}, avg test ppl: {sum(all_ppls) / len(all_ppls)}')


if __name__ == '__main__':
    device = 'cuda:0'
    tokenizer = setup_tokenizer()
    config = OmegaConf.load('configs/base.yaml')
    train_data = load_wikidata('data/wikitext-103/train.txt')
    test_data = load_wikidata('data/wikitext-103/test.txt')
    train_eval(train_data, test_data, tokenizer, config)

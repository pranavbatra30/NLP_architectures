!pip install torchtext==0.9.0
!pip install -U spacy
!python -m spacy download en_core_web_sm
!python -m spacy download de_core_news_sm

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.legacy.data import Field, BucketIterator
from torch.nn import functional as F
import numpy as np
import spacy
from torchtext.datasets import IWSLT2016
from torchtext.legacy.datasets import TranslationDataset

# Load the spacy models for tokenization
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

# Tokenization functions
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def create_masks(src, trg):
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    trg_mask = (trg != TRG.vocab.stoi['<pad>']).unsqueeze(-2)

    trg_len = trg.size(1)
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
    trg_mask = trg_mask & trg_sub_mask

    return src_mask, trg_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        self.attn = attn
        attn = self.dropout(attn)
        output = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        output = self.W_o(output)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.W_1(x)))
        x = self.W_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, trg_mask, src_mask):
        self_attn_output = self.self_attn(x, x, x, trg_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# Hyperparameters
d_model = 256
num_heads = 4
num_layers = 3
d_ff = 512
dropout = 0.1
epochs = 20
batch_size = 64
learning_rate = 0.0005

# Preprocessing
SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

train_data, valid_data, test_data = IWSLT2016(
    split=('train', 'valid', 'test'),
    language_pair=('de', 'en'),
    root='.data',
    fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    device=device,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src))

# Encoder and decoder classes
class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, src, src_mask):
        x = self.embedding(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, num_layers, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, trg, enc_output, trg_mask, src_mask):
        x = self.embedding(trg)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc_output, trg_mask, src_mask)
        x = self.fc_out(x)
        return x

# Transformer class
class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, src_mask, trg_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, trg_mask, src_mask)
        return dec_output

input_dim = len(SRC.vocab)
output_dim = len(TRG.vocab)

enc = Encoder(input_dim, d_model, num_layers, num_heads, d_ff, dropout)
dec = Decoder(output_dim, d_model, num_layers, num_heads, d_ff, dropout)

model = Transformer(enc, dec).to(device)

# Training and evaluation functions
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, trg = batch.src, batch.trg
        src_mask, trg_mask = create_masks(src, trg)

        optimizer.zero_grad()

        output = model(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])

        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch.src, batch.trg
            src_mask, trg_mask = create_masks(src, trg)

            output = model(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])

            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])

# Training loop
for epoch in range(epochs):
    train_loss = train(model, train_iterator, optimizer, criterion, clip=1)
    valid_loss = evaluate(model, valid_iterator, criterion)
    print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}")

# Evaluate on test set
test_loss = evaluate(model, test_iterator, criterion)
print(f"Test Loss: {test_loss:.3f}")
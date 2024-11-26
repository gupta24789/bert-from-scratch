import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.d_model = config['d_model']
        self.max_len = config['max_len']
        self.dropout = nn.Dropout(config['dropout'])

        ## create matrix of shape (max_len, d_model)
        pe = torch.zeros(self.max_len, self.d_model)
        pe.requires_grad_(False)
        ## create vector of shape (max_len, 1)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1) 
        ## create vector of shape (d_model/2)       
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0)/self.d_model))
        ## apply sin to even pos
        pe[:, 0::2] = torch.sin(position * div_term)
        ## apply cos to odd pos
        pe[:, 1::2] = torch.cos(position * div_term)
        ## Add batch dim to position encoding (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, config, vocab_size):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        # (batch, max_len) --> (batch, max_len, d_model)
        self.token = torch.nn.Embedding(vocab_size, config['d_model'], padding_idx=0)
        self.segment = torch.nn.Embedding(3, config['d_model'], padding_idx=0)
        self.position = PositionalEmbedding(config)
        self.dropout = torch.nn.Dropout(p= config['dropout'])
       
    def forward(self, sequence, segment_label):
        # (batch, max_len, d_model)
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return x
    
class MultiHeadAttentionBlock(nn.Module):
    """
    This is fast implementation of Attention head
    """
    def __init__(self,config) -> None:
        super().__init__()
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.max_len = config['max_len']
        assert self.d_model % self.n_heads == 0, "d_model is not divisible by h"

        self.d_k = self.d_model // self.n_heads # Dimension of vector seen by each head
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False) # Wq
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False) # Wk
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False) # Wv
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False) # Wo
        self.dropout = nn.Dropout(config['dropout'])

        mask = torch.tril(torch.ones((1, 1, 1, self.max_len), dtype= torch.long))
        self.register_buffer("mask", mask)

    def attention(self, query, key, value, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, max_len, d_k) --> (batch, h, max_len, max_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        attention_scores.masked_fill_(self.mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, max_len, max_len) # Apply softmax
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, max_len, max_len) --> (batch, h, max_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v):
        query = self.w_q(q) # (batch, max_len, d_model) --> (batch, max_len, d_model)
        key = self.w_k(k) # (batch, max_len, d_model) --> (batch, max_len, d_model)
        value = self.w_v(v) # (batch, max_len, d_model) --> (batch, max_len, d_model)

        # (batch, max_len, d_model) --> (batch, max_len, h, d_k) --> (batch, h, max_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = self.attention(query, key, value, self.dropout)
        
        # Combine all the heads together
        # (batch, h, max_len, d_k) --> (batch, max_len, h, d_k) --> (batch, max_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        # Multiply by Wo
        # (batch, max_len, d_model) --> (batch, max_len, d_model)  
        return self.w_o(x)

class FeedForwardBlock(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.d_model = config['d_model']
        self.hidden_dim = config['hidden_dim']
        self.dropout = config['dropout']
        self.linear_1 = nn.Linear(self.d_model, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.linear_2 = nn.Linear(self.hidden_dim, self.d_model)

    def forward(self, x):
        # (batch, max_len, d_model) -> (batch, max_len, d_model)
        return self.linear_2(self.dropout(F.gelu(self.linear_1(x))))
    
class LayerNorm(nn.Module):

    def __init__(self, features: int, eps = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) ## multiple
        self.bias = nn.Parameter(torch.zeros(features)) ## add

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class ResidualBlock(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.norm = LayerNorm(config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.self_attention_block = MultiHeadAttentionBlock(config)
        self.feed_forward_block = FeedForwardBlock(config)
        self.residual_connection = nn.ModuleList([
            ResidualBlock(config) for i in range(2)
        ])

    def forward(self, x):
        # (batch_size, max_len, d_model)
        ## residual with attention block
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x))
        ## residual with feedforward block
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config['n_layers'])
        ])
        self.norm = LayerNorm(config['d_model'])

    def forward(self, x):
        for layer in self.encoder_layers:
            x= layer(x)
        return self.norm(x)
    
class NextSentencePrediction(torch.nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """
    def __init__(self, d_model):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = torch.nn.Linear(d_model, 2)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # use only the first token which is the [CLS]
        return self.softmax(self.linear(x[:, 0, :]))
    
class MaskedLanguageModel(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
    
class BERTModel(nn.Module):

    def __init__(self, config, vocab_size) -> None:
        super().__init__()
        self.d_model = config['d_model']
        self.vocab_size = vocab_size
        self.config = config
        self.bert_emb = BERTEmbedding(config, vocab_size)
        self.encoder = Encoder(config)

    def forward(self, x, segment_label):
        emb = self.bert_emb(x, segment_label)
        encoder_out = self.encoder(emb)    #(batch, max_len, d_model)
        # print(f"Encoder Out : {encoder_out.shape}")
        return encoder_out

class BERTLM(nn.Module):

    def __init__(self, bert : BERTModel, vocab_size : int) -> None:
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)


    def forward(self, bert_input, segment_input):
        x = self.bert(bert_input, segment_input)
        return self.next_sentence(x), self.mask_lm(x)
    





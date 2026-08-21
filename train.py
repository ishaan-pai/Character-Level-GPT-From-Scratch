# training on tinyshakespeare data set available at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337) # number isn't necessary, just important for recreating the tutorial

# opening input file and reading entire contents
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# creating a list of characters that appear in the text. In other words, these are all the characters the LLM should need to use.
chars = sorted(list(set(text)))
vocab_size = len(chars)

#testing previous section
"""
print(''.join(chars))
print(vocab_size)
"""

# tokenization and converting from characters to integers using mapping
stoi = {}
for ch, i in enumerate(chars):
    stoi[i] = ch
itos = {}
for i, ch in enumerate(chars):
    itos[i] = ch
encode = lambda s: [stoi[c] for c in s] # take a string and output a list of integers based on that string
decode = lambda l: ''.join([itos[i] for i in l]) # take a list of integers and output a string based on those integers

# testing previous section
"""
print(encode("Hello World"))
print(decode(encode("Hello World")))
"""

# creating tensor based on text
data = torch.tensor(encode(text), dtype=torch.long)

#testing previous section
"""
print(data.shape, data.dtype)
print(data[:1000])
"""

# splitting up data into training and validation sets
n = int(0.9 * len(data)) # first 90% of text will be training, the rest will be validation
train_data = data[:n]
val_data = data[n:]

# chunking
batch_size = 32 # how many sequences are we willing to process in parallel?
block_size = 8 # max context length for predictions

def get_batch(split):
    # generate a batch of data with inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

# testing for previous section
"""
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

print("------")

for b in range(batch_size): #batch dimensions
    for t in range(block_size): #time dimensions
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target is: {target}")
"""

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (Batch x Time) tensor of integers
        logits = self.token_embedding_table(idx) # (Batch x Time x Channel)

        if targets is None:
            loss = None
        else:
            # logits reshaping to match F.cross_entropy() format
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (Batch x Time) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (Batch x Channel)
            #a apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1) # Batch x Channel
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # Batch x 1
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # Batch x Time + 1
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)

# test
"""
print(logits.shape)
print(loss)
print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
"""

# pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# warning: be careful changing value in range for training. It makes the accuracy better (lower loss.item()) however, can definitely force the cpu into some effort
for steps in range(100000):

    # sample batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))




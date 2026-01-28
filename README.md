# One-Step LanguageGeneration, Exact Likelihood with Optimal Transport

A two-stage framework for discrete sequence generation that combines normalizing flows with direct one-step generators.

## Core Idea

Traditional generative models for discrete data require expensive autoregressive or iterative sampling. 
We propose a generation framework that maps the noise to the data space in one step. The idea is first using Normalizing Flow Encoder to encode the data to the latent space, which creates a deterministic mapping (optimal transport in some sense/metrics that is easy for NN to learn, not quite clear, wanna categorize this in a more precise language) from the data space to the latent space. In the stage B, we train a One-Step Generator (Any architectures, such as BERT) to map the noise to the data space in one forward pass.


We propose a two-stage training approach that decouples encoding from generation:

### Stage A: Train a Normalizing Flow Encoder (X → Z)

We train an encoder that maps discrete sequences to a Gaussian latent space:

```
X (discrete sequence) → TextEncoder → U (embeddings) → TarFlow → Z ~ N(0, I)
```

The encoder is trained with the standard flow objective (negative log-likelihood):

```
L_flow = -log p(x) = 0.5 * ||z||² - log|det(∂z/∂x)|
```

This gives us a deterministic mapping from any sequence X to a corresponding latent Z.

### Stage B: Train a One-Step Generator (Z → X)

Once we have a trained encoder, we can create paired data (X, Z) by encoding our training set. We then train a simple feedforward generator (e.g., BERT-style transformer) to directly map Z back to X:

```
Z ~ N(0, I) → Generator → logits → X̂
```

**Training Losses:**
1. **Reconstruction**: Cross-entropy between predicted logits and true tokens
2. **Representation Alignment** (optional): MSE between generator and encoder hidden states

**Key Insight**: The generator doesn't need to be invertible or autoregressive. It's a simple forward network trained with standard classification loss, making it:
- **Fast**: One-step generation (no sequential sampling)
- **Flexible**: Any architecture works (BERT, MLP, etc.)
- **Trainable**: Simple cross-entropy loss, no complex flow training

### Generation

At inference time, sampling is trivial:
```python
z = torch.randn(batch_size, seq_len, hidden_dim)  # Sample from Gaussian
logits = generator(z)                               # One forward pass
tokens = logits.argmax(dim=-1)                      # Decode to tokens
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STAGE A (Encoder)                        │
│  X (one-hot) → TextEncoder → U → TarFlow → Z ~ N(0,I)          │
│                    ↓              ↓                              │
│              [h₁, h₂, ...]   [h₃, h₄, ...]  (hidden states)     │
└─────────────────────────────────────────────────────────────────┘
                              ↓ freeze
┌─────────────────────────────────────────────────────────────────┐
│                       STAGE B (Generator)                       │
│  Z → Generator (BERT) → logits → X̂                             │
│         ↓                                                        │
│    [g₁, g₂, ...] ←──align──→ [h₁, h₂, ...]                     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Advantages

1. **Exact Likelihood**: not quite clear but bc NFencoder training doesnt use any ELBO, but use MLE, so I think the generator has exact likelihood with the classfication loss.

2. **One-Step Sampling**: Unlike autoregressive flows that require O(T) sequential steps, our generator produces all tokens in parallel.

3. **Flexible Generator**: The generator can be any architecture (BERT, GPT, MLP) since it only needs to minimize reconstruction loss.

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    The configuration class for the GPT model.
    
    Parameters:
    - context_size (int): The size of the context window. Default is 1024.
    - vocab_size (int): The size of the vocabulary. Default is 50257.
    - n_blocks (int): The number of blocks in the model. Default is 12.
    - n_heads (int): The number of heads in the model. Default is 12.
    - n_embed (int): The size of the embeddings. Default is 768.
    """
    
    context_size: int = 1024
    vocab_size: int = 50257
    n_blocks: int = 12
    n_heads: int = 12
    n_embed: int = 768
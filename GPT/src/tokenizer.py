import tiktoken


class Tokenizer:
    
    ### Magic methods ###
    
    def __init__(self, tokenizer: str = "gpt2") -> None:
        """
        Initialize the tokenizer.
        
        Parameters:
        - tokenizer (str): The name of the tokenizer. Default is "gpt2".
        """
        
        # Get the tokenizer
        self.tokenizer = tiktoken.get_encoding(tokenizer)
        
    ### Public methods ###
    
    def encode(self, text: str) -> list[int]:
        """
        Method that encodes the text into tokens.
        
        Parameters:
        - text (str): The text to encode.
        
        Returns:
        - list[int]: The encoded tokens.
        """
        
        return self.tokenizer.encode(text)
    
    
    def decode(self, tokens: list[int]) -> str:
        """
        Method that decodes the tokens into text.
        
        Parameters:
        - tokens (list[int]): The tokens to decode.
        
        Returns:
        - str: The decoded text.
        """
        
        return self.tokenizer.decode(tokens)
    
    
    def vocab_size(self) -> int:
        """
        Method that returns the vocabulary size.
        
        Returns:
        - int: The vocabulary size.
        """
        
        return self.tokenizer.n_vocab
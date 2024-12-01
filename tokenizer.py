# Sample tokenizer for Greek epics
import re

class WordTokenizer:
    def __init__(self):
        self.vocab = []
        self.vocab_size = 0
        self.string_to_int = {}
        self.int_to_string = {}


    def create_vocab(self, text):
        self.vocab = list(sorted(set(re.findall(r'\w+|\S', text)))) 
        self.vocab_size = len(self.vocab)
        self.string_to_int = {character: integer for integer, character in enumerate(self.vocab)}
        self.int_to_string = {integer: character for integer, character in enumerate(self.vocab)}
        return self.vocab


    def encode(self, text):
        processed_text = re.findall(r'\w+|\S', text)
        return [self.string_to_int[token] for token in processed_text] 
    

    def decode(self, text):
        #processed_text = re.findall(r'\w+|\S', text)
        return [self.int_to_string[token] for token in text]

# Testing 

if __name__ == '__main__':

    with open('shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()


    # Testing tokenizer

    test_text = "To be, or not to be: that is the question."
    tokenizer = WordTokenizer()
    vocab = tokenizer.create_vocab(test_text)

    # Test create_vocab
    print("Vocabulary:", vocab[:10])
    assert len(vocab) > 0, "Vocabulary should not be empty"
    assert "be" in vocab, "'be' should be in vocabulary"

    # Test encoding
    encoded_text = tokenizer.encode(test_text)
    print("Encoded:", encoded_text)
    assert isinstance(encoded_text, list), "Encoded output should be a list of integers"
    assert len(encoded_text) == len(re.findall(r'\w+|\S', test_text)), "Encoded output length should match token count"

    # Test decoding
    decoded_text = tokenizer.decode(encoded_text)
    print("Decoded:", decoded_text)
    assert ' '.join(decoded_text) == ' '.join(re.findall(r'\w+|\S', test_text)), "Decoded text should match tokenized text"

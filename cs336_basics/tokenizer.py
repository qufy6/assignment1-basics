# Tokenizer Class
from typing import Dict, Tuple, List, Iterable, Iterator
import regex as re
import pickle
from tqdm import trange, tqdm

def text2chunks(text, special_tokens):
    special_tokens =  sorted(special_tokens, key = len, reverse = True)
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(t) for t in special_tokens)
    Chunks = re.split(f"({pattern})", text)
    return Chunks

def chunk2tokens(chunk):
    PAT = (r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    Tokens = re.findall(PAT, chunk)
    return Tokens

def token2bytes(token):
    Bytes = [bytes([b]) for b in token.encode("utf-8")]
    return Bytes



class Tokenizer:
    def __init__(
        self, 
        vocab: Dict[int, bytes], 
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = []
    ):
        self.vocab = vocab
        self.merges = merges
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        
        if special_tokens is None:
            self.special_tokens = {}
            self.bytes_special_tokens = []
        else:
            self.special_tokens = {token: i for i, token in enumerate(special_tokens, start=len(self.vocab))}
            self.bytes_special_tokens = [token.encode("utf-8") for token in special_tokens if isinstance(token, str)]
        
    
    def encode(self, text: str) -> List[int]:
        token_list = []
        chunks = text2chunks(text, list(self.special_tokens.keys()))
        for chunk in chunks:
            if chunk in self.special_tokens.keys():
                token_list.append(chunk.encode("utf-8"))
            else:
                tokens = chunk2tokens(chunk)
                for token in tokens:
                    bytes_per_token = token2bytes(token)
                    token_list.append(bytes_per_token)
                    
        token_ids = []
        for byte_token in token_list:
            if byte_token in self.bytes_special_tokens:
                token_ids.append([self.vocab_inv[byte_token]])
            else:
                token_ids.append([self.vocab_inv[b] for b in byte_token]) 

        for i, pretoken in enumerate(token_ids):
            for merge in self.merges:
                new_index = self.vocab_inv.get(merge[0] + merge[1], None)
                if new_index is None:
                    continue
                merged = []
                j = 0
                while j < len(pretoken):
                    if (
                        j < len(pretoken) - 1
                        and (self.vocab[pretoken[j]], self.vocab[pretoken[j + 1]]) == merge
                    ):
                        merged.append(new_index)
                        j += 2
                    else:
                        merged.append(pretoken[j])
                        j += 1
                        
                pretoken = merged
            token_ids[i] = pretoken[:]

        return [i for pre in token_ids for i in pre]
    

    def encode_iterable(self, iterable: Iterable[str], batch_size: int = 1024) -> Iterator[int]:
        batch = []
        for line in tqdm(iterable):
            if not line:
                continue
            batch.append(line)
            if len(batch) >= batch_size:
                for encoded in map(self.encode, batch):
                    yield from encoded
                batch.clear()
                
        if batch:
            for encoded in map(self.encode, batch):
                yield from encoded
    
    def decode(self, ids: list[int]) -> str:
        # https://en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character -- U+FFFD is b"\xef\xbf\xbd"
        
        tokens = b"".join(self.vocab.get(i, b"\xef\xbf\xbd") for i in ids)
        return tokens.decode("utf-8", errors="replace") #Replacement_character
    
    @classmethod
    def from_files(
        cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None
    ):
        with open(vocab_path, 'rb') as vf:
            raw_vocab = pickle.load(vf)

        vocab = {int(k): (v.encode("utf-8") if isinstance(v, str) else v)
                for k, v in raw_vocab.items()}

        with open(merges_path, 'rb') as mf:
            raw_merges = pickle.load(mf)

        merges = []
        for a, b in raw_merges:
            merges.append((
                a.encode("utf-8") if isinstance(a, str) else a,
                b.encode("utf-8") if isinstance(b, str) else b
            ))
        return cls(vocab, merges, special_tokens)
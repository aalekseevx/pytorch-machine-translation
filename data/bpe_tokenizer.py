from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


class BPETokenizer:
    def __init__(self, sentence_list, vocab_size, max_length):
        """
        sentence_list - список предложений для обучения
        """
        self.max_length = max_length
        self.special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]

        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=self.special_tokens)
        self.tokenizer.train_from_iterator(sentence_list, trainer=trainer)
        self.tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[
                ("[SOS]", self.tokenizer.token_to_id("[SOS]")),
                ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
            ]
        )

    def do_pad(self, inp):
        self.max_length = self.max_length
        if len(inp) < self.max_length:
            return inp + [self.tokenizer.token_to_id("[PAD]")] * (self.max_length - len(inp))
        else:
            return inp[:self.max_length - 1] + [self.tokenizer.token_to_id("[EOS]")]

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        return self.do_pad(self.tokenizer.encode(sentence).ids)

    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        decoded = self.tokenizer.decode(token_list)
        return [token for token in decoded.split() if token not in self.special_tokens]

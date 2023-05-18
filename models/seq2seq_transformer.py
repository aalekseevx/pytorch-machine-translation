import torch
import torch.nn as nn
import metrics

import numpy as np

from models.positional_encoding import PositionalEncoding


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(
        self,
        device,
        emb_size,
        vocab_size,
        max_seq_len,
        target_tokenizer,
        lr,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward=500,
        step_size=10,
        gamma=0.3,
    ):
        super(Seq2SeqTransformer, self).__init__()

        self.device = device
        self.emb_size = emb_size

        self.target_tokenizer = target_tokenizer

        self.embedding = nn.Embedding(vocab_size, emb_size).to(device)
        self.pos_encoder = PositionalEncoding(emb_size, max_seq_len).to(device)
        self.decoder = nn.Linear(emb_size, vocab_size).to(device)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        print('d_model', emb_size)
        self.model = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        ).to(device)

    def forward(self, batch):
        inp, target = batch

        inp = self.pos_encoder(self.embedding(inp).long() * np.sqrt(self.emb_size))
        target = self.pos_encoder(self.embedding(target).long() * np.sqrt(self.emb_size))

        inp = np.transpose(inp, (1, 0, 2))
        target = np.transpose(target, (1, 0, 2))

        # print(inp.shape)
        # print(target.shape)

        model_output = self.model(
            inp,
            target,
            self.model.generate_square_subsequent_mask(inp.size(0), device=self.device),
            self.model.generate_square_subsequent_mask(inp.size(0), device=self.device),
        )
        decoder_output = self.decoder(model_output)
        decoder_output = decoder_output.permute(1, 0, 2)

        return torch.argmax(decoder_output, dim=-1).clone(), decoder_output

    def training_step(self, batch):
        self.optimizer.zero_grad()
        _, model_output = self.forward(batch)
        inp, target = batch

        # print(model_output.shape)
        # print(target.shape)

        loss = self.loss(model_output.reshape(-1, model_output.shape[-1]), target.reshape(-1))

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self, batch):
        inp, target = batch
        _, model_output = self.forward(batch)
        loss = self.loss(model_output.reshape(-1, model_output.shape[-1]), target.reshape(-1))
        return loss.item()

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = predicted.squeeze(-1).detach().cpu().numpy()
        actuals = target_tensor.squeeze(-1).detach().cpu().numpy()
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences

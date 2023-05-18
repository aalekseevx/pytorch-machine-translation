import torch
import yaml
from models import trainer
from data.datamodule import DataManager
from txt_logger import TXTLogger
from models.seq2seq_transformer import Seq2SeqTransformer

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    data_config = yaml.load(open("configs/data_config.yaml", 'r'),   Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader = dm.prepare_data()

    model_config = yaml.load(open("configs/model_config.yaml", 'r'),   Loader=yaml.Loader)

    model = Seq2SeqTransformer(
        device=DEVICE,
        emb_size=model_config['emb_size'],
        vocab_size=4000,
        max_seq_len=30,
        target_tokenizer=dm.target_tokenizer,
        lr=model_config['lr'],
        nhead=model_config['nhead'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
    )

    logger = TXTLogger('training_logs')
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger)

    if model_config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(dev_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)





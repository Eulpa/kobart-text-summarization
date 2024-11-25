import argparse
import numpy as np
import pandas as pd

from loguru import logger

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from dataset import KobartSummaryModule
from model import KoBARTConditionalGeneration

from transformers import PreTrainedTokenizerFast

parser = argparse.ArgumentParser(description='KoBART Summarization')

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='data/train.tsv',
                            help='train file')
        parser.add_argument('--test_file',
                            type=str,
                            default='data/test.tsv',
                            help='test file')
        parser.add_argument('--batch_size',
                            type=int,
                            default=28,
                            help='')
        parser.add_argument('--checkpoint',
                            type=str,
                            default='checkpoint',
                            help='')
        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        parser.add_argument('--max_epochs',
                            type=int,
                            default=10,
                            help='train epochs')
        parser.add_argument('--lr',
                            type=float,
                            default=2e-6,
                            help='The initial learning rate')
        parser.add_argument('--accelerator',
                            type=str,
                            default='gpu',
                            choices=['gpu', 'cpu'],
                            help='select accelerator')
        parser.add_argument('--num_gpus',
                            type=int,
                            default=1,
                            help='number of gpus')
        parser.add_argument('--gradient_clip_val',
                            type=float,
                            default=1.5,
                            help='gradient_clipping')
        parser.add_argument('--ckpt_path', 
                            type=str, 
                            default=None, 
                            help='Path to a checkpoint file to resume training')
        parser.add_argument('--run_name',
                            type=str,
                            default=None,
                            help='Wandb run name')
        parser.add_argument('--run_id',
                            type=str,
                            default=None,
                            help='Wandb run id')


        return parser


if __name__ == '__main__':
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KobartSummaryModule.add_model_specific_args(parser)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    args = parser.parse_args()
    logger.info(args)
    
    dm = KobartSummaryModule(args.train_file,
                        args.test_file,
                        tokenizer,
                        batch_size=args.batch_size,
                        max_len=args.max_len,
                        num_workers=args.num_workers)
    dm.setup("fit")
    
    model = KoBARTConditionalGeneration(args)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=args.checkpoint,
                                          filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                          verbose=True,
                                          save_last=True,
                                          mode='min',
                                          save_top_k=3,
                                          )
    
    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            patience=5,
                                            verbose=True,
                                            mode='min'
                                            )
    
    wandb_logger = WandbLogger(project="KoBART-summ",
                               name=args.run_name,
                               id=args.run_id,
                               resume="allow")

    trainer = L.Trainer(max_epochs=args.max_epochs,
                        accelerator=args.accelerator,
                        devices=args.num_gpus,
                        gradient_clip_val=args.gradient_clip_val,
                        callbacks=[checkpoint_callback, early_stopping_callback],
                        logger=wandb_logger,
                        )
    
    trainer.fit(model, dm, ckpt_path=args.ckpt_path)

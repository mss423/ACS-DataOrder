# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:19:30 2020

@author: Jiang Yuxin
"""

import torch
from torch import nn
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer
)
     
        
class BertModel(nn.Module):
    def __init__(self, num_labels, requires_grad = True):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            'google-bert/bert-base-uncased',
            num_labels=num_labels,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            problem_type="multi_label_classification" if num_labels > 2 else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        outputs = self.bert(
            input_ids=batch_seqs.to(self.device), # move inputs to device
            attention_mask=batch_seq_masks.to(self.device),
            token_type_ids=batch_seq_segments.to(self.device),
            labels=labels.to(self.device)
        )
        loss = outputs.loss
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
        return loss, logits, probabilities
        
        
        

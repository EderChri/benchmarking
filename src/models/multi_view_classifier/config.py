from typing import Optional
import torch.nn as nn
from merlion.models.anomaly.base import DetectorConfig
import numpy as np
import pandas as pd


class MultiViewClassifierConfig(DetectorConfig):
    """Configuration for Multi-View Transformer Classifier."""
    
    def __init__(
        self,
        # Model architecture
        num_feature: int = 1,
        num_embedding: int = 128,
        num_hidden: int = 256,
        num_head: int = 8,
        num_layers: int = 2,
        num_target: int = 2,
        dropout: float = 0.1,
        
        # Training parameters
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        l1_scale: float = 0.0,
        l2_scale: float = 0.01,
        
        # Loss configuration
        loss_type: str = 'ALL',
        feature: str = 'latent',
        temperature: float = 0.5,
        lam: float = 0.1,
        
        # Training mode
        mode: str = 'finetune',
        
        # Data augmentation
        augmentation_strength: float = 0.1,
        
        # Device
        use_gpu: bool = True,
        
        # Pre-trained weights
        checkpoint_path: Optional[str] = None,
        
        # DetectorConfig parameters
        max_score: float = 1.0,
        threshold=None,
        enable_calibrator: bool = True,
        enable_threshold: bool = True,
        transform=None,
        **kwargs
    ):
        self.num_feature = num_feature
        self.num_embedding = num_embedding
        self.num_hidden = num_hidden
        self.num_head = num_head
        self.num_layers = num_layers
        self.num_target = num_target
        self.dropout = dropout
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_scale = l1_scale
        self.l2_scale = l2_scale
        
        self.loss_type = loss_type
        self.feature = feature
        self.temperature = temperature
        self.lam = lam
        
        self.mode = mode
        self.augmentation_strength = augmentation_strength
        self.use_gpu = use_gpu
        self.checkpoint_path = checkpoint_path
        
        super().__init__(
            max_score=max_score,
            threshold=threshold,
            enable_calibrator=enable_calibrator,
            enable_threshold=enable_threshold,
            transform=transform,
            **kwargs
        )

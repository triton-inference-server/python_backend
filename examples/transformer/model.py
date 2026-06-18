import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import triton_python_backend_utils as pb_utils
import random


class SentimentClassifier(nn.Module):
    """
    A transformer-based sentiment classifier model.
    This model takes tokenized text sequences as input and outputs sentiment scores.
    """
    def __init__(self, vocab_size=10000, embed_dim=256, num_heads=8, 
                 num_layers=4, max_seq_length=128, num_classes=3):
        super(SentimentClassifier, self).__init__()
        """
        Initialize the sentiment classifier model.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_seq_length: Maximum sequence length
            num_classes: Number of sentiment classes
        """
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(embed_dim // 2, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        embeddings = token_embeds + position_embeds
        
        # Create attention mask for transformer (inverted: 1 -> can attend, 0 -> cannot attend)
        if attention_mask is not None:
            # Convert to boolean mask (True = masked position)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        encoded = self.transformer_encoder(
            embeddings, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Pool: take the mean of all non-padded tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(encoded.size()).float()
            sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = encoded.mean(dim=1)
        
        # Classification
        x = self.fc1(pooled)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class TritonPythonModel:
    """
    Triton Python backend model wrapper for the transformer sentiment classifier.
    """
    def initialize(self, args):
        """
        Initialize the model. This function is called once when the model is loaded.
        
        Args:
            args: Dictionary containing initialization parameters
        """
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get output configuration
        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])
        
        # Model parameters
        self.vocab_size = 10000
        self.embed_dim = 256
        self.num_heads = 8
        self.num_layers = 4
        self.max_seq_length = 128
        self.num_classes = 3  # Negative, Neutral, Positive
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set all random seeds for reproducibility BEFORE model initialization
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Initialize the model
        self.model = SentimentClassifier(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_length=self.max_seq_length,
            num_classes=self.num_classes
        ).to(self.device)
        
        # Set model to evaluation mode to disable dropout and ensure consistent outputs
        self.model.eval()
        
        print(f"Model initialized on device: {self.device}")
        print(f"Model parameters: vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, "
              f"num_heads={self.num_heads}, num_layers={self.num_layers}, "
              f"max_seq_length={self.max_seq_length}, num_classes={self.num_classes}")
    
    def execute(self, requests):
        """
        Execute inference on a batch of requests.
        
        Args:
            requests: List of pb_utils.InferenceRequest objects
            
        Returns:
            List of pb_utils.InferenceResponse objects
        """
        responses = []
        
        for request in requests:
            # Get input tensors
            input_ids_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IDS")
            attention_mask_tensor = pb_utils.get_input_tensor_by_name(request, "ATTENTION_MASK")
            
            # Convert to numpy
            input_ids_np = input_ids_tensor.as_numpy()
            attention_mask_np = attention_mask_tensor.as_numpy()
            
            # Convert to torch tensors
            input_ids = torch.from_numpy(input_ids_np).long().to(self.device)
            attention_mask = torch.from_numpy(attention_mask_np).long().to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
            
                # Apply softmax to get probabilities
                probabilities = F.softmax(logits, dim=-1)
            
            # Convert output to numpy
            output_np = probabilities.cpu().numpy().astype(self.output_dtype)
            
            # Create output tensor
            output_tensor = pb_utils.Tensor("OUTPUT", output_np)
            
            # Create inference response
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
        
        return responses
    
    def finalize(self):
        """
        Clean up resources when the model is unloaded.
        """
        print("Cleaning up model resources...")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


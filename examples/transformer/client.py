#!/usr/bin/env python3
import numpy as np
import sys
import argparse
from typing import List, Tuple
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class SimpleTokenizer:
    """
    A simple character-level tokenizer for demo purposes.
    """
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        
    def encode(self, text: str, max_length: int = 128) -> Tuple[List[int], List[int]]:
        """
        Encode text to token IDs and create attention mask.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        # Simple character-level encoding that maps each character to an ID based on its ASCII value
        input_ids = [min(ord(c), self.vocab_size - 1) for c in text.lower()]
        
        # Truncate if too long
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        padding_length = max_length - len(input_ids)
        input_ids.extend([self.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        return input_ids, attention_mask


class SentimentClient:
    """
    Client for the Transformer Sentiment Classifier on Triton Inference Server.
    """
    
    def __init__(self, url: str = "localhost:8000", model_name: str = "transformer"):
        """
        Initialize the client.
        
        Args:
            url: Triton server URL (e.g., "localhost:8000")
            model_name: Name of the model 
        """
        self.url = url
        self.model_name = model_name
        self.client = httpclient.InferenceServerClient(url=url, verbose=False)
        self.tokenizer = SimpleTokenizer()
        self.max_seq_length = 128
        self.class_names = ["Negative", "Neutral", "Positive"]
        
    def check_server_ready(self) -> bool:
        """Check if the Triton server is ready."""
        try:
            if self.client.is_server_ready():
                print(f"Server at {self.url} is ready")
                return True
            else:
                print(f"Server at {self.url} is not ready")
                return False
        except InferenceServerException as e:
            print(f"Failed to connect to server at {self.url}")
            print(f"  Error: {e}")
            return False
    
    def check_model_ready(self) -> bool:
        """Check if the model is ready."""
        try:
            if self.client.is_model_ready(self.model_name):
                print(f"Model '{self.model_name}' is ready")
                return True
            else:
                print(f"Model '{self.model_name}' is not ready")
                return False
        except InferenceServerException as e:
            print(f"Failed to check model status")
            print(f"  Error: {e}")
            return False
    
    def predict(self, text: str) -> Tuple[np.ndarray, int, str]:
        """
        Run inference on a single text input.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (probabilities, predicted_class, class_name)
        """
        # Tokenize input
        input_ids, attention_mask = self.tokenizer.encode(text, self.max_seq_length)
        
        # Convert to numpy arrays with batch dimension
        input_ids_np = np.array([input_ids], dtype=np.int64)
        attention_mask_np = np.array([attention_mask], dtype=np.int64)
        
        # Create input objects
        inputs = [
            httpclient.InferInput("INPUT_IDS", input_ids_np.shape, "INT64"),
            httpclient.InferInput("ATTENTION_MASK", attention_mask_np.shape, "INT64")
        ]
        
        # Set data
        inputs[0].set_data_from_numpy(input_ids_np)
        inputs[1].set_data_from_numpy(attention_mask_np)
        
        # Create output object
        outputs = [httpclient.InferRequestedOutput("OUTPUT")]
        
        # Send inference request
        try:
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            # Get output
            output = response.as_numpy("OUTPUT")[0]  # Remove batch dimension
            predicted_class = int(np.argmax(output))
            class_name = self.class_names[predicted_class]
            
            return output, predicted_class, class_name
            
        except InferenceServerException as e:
            print(f"Inference failed: {e}")
            raise
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[np.ndarray, int, str]]:
        """
        Run inference on a batch of text inputs.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of tuples (probabilities, predicted_class, class_name) for each input
        """
        # Tokenize all inputs
        input_ids_batch = []
        attention_mask_batch = []
        
        for text in texts:
            input_ids, attention_mask = self.tokenizer.encode(text, self.max_seq_length)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
        
        # Convert to numpy arrays
        input_ids_np = np.array(input_ids_batch, dtype=np.int64)
        attention_mask_np = np.array(attention_mask_batch, dtype=np.int64)
        
        # Create input objects
        inputs = [
            httpclient.InferInput("INPUT_IDS", input_ids_np.shape, "INT64"),
            httpclient.InferInput("ATTENTION_MASK", attention_mask_np.shape, "INT64")
        ]
        
        # Set data
        inputs[0].set_data_from_numpy(input_ids_np)
        inputs[1].set_data_from_numpy(attention_mask_np)
        
        # Create output object
        outputs = [httpclient.InferRequestedOutput("OUTPUT")]
        
        # Send inference request
        try:
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            # Get outputs
            outputs_np = response.as_numpy("OUTPUT")
            
            results = []
            for output in outputs_np:
                predicted_class = int(np.argmax(output))
                class_name = self.class_names[predicted_class]
                results.append((output, predicted_class, class_name))
            
            return results
            
        except InferenceServerException as e:
            print(f"Batch inference failed: {e}")
            raise


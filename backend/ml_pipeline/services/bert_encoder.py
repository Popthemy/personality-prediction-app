"""
BERT Contextual Embedding Extraction Service.

Uses HuggingFace transformers bert-base-uncased to extract 768-dimensional contextual embeddings.
"""
import logging
import torch
from typing import List, Dict
import json

logger = logging.getLogger('ml_pipeline')

# Global model cache
_bert_model = None
_bert_tokenizer = None


def get_bert_model_and_tokenizer():
    """Lazy load BERT model and tokenizer."""
    global _bert_model, _bert_tokenizer
    
    if _bert_model is None or _bert_tokenizer is None:
        from transformers import AutoTokenizer, AutoModel
        
        logger.info("Loading BERT model (bert-base-uncased)...")
        _bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        _bert_model = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _bert_model = _bert_model.to(device)
        _bert_model.eval()
        
        logger.info(f"BERT loaded on device: {device}")
    
    return _bert_model, _bert_tokenizer


class BERTEncoder:
    """BERT embedding encoder."""
    
    def __init__(self):
        self.model, self.tokenizer = get_bert_model_and_tokenizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def encode_text(self, text: str, max_length: int = 512) -> Dict:
        """
        Encode text to 768-dimensional BERT embedding.
        
        Uses the [CLS] token embedding as the sentence representation.
        
        Args:
            text: Text to encode
            max_length: Maximum token length
        
        Returns:
            Dict with 'embedding' (768-dim) and metadata
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors='pt',
            padding=True
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Extract [CLS] token embedding (first token, last hidden state)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        
        # Convert to list for JSON serialization
        embedding_list = cls_embedding.tolist()
        
        return {
            'embedding': embedding_list,
            'embedding_dim': len(embedding_list),
            'model_name': 'bert-base-uncased',
            'text_length': len(text),
            'tokens_count': len(input_ids[0]),
        }
    
    def encode_batch(self, texts: List[str], max_length: int = 512) -> List[Dict]:
        """
        Encode multiple texts efficiently.
        
        Args:
            texts: List of texts
            max_length: Maximum token length
        
        Returns:
            List of embedding dicts
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.encode_text(text, max_length)
                results.append(result)
                logger.debug(f"Encoded text {i+1}/{len(texts)}")
            except Exception as e:
                logger.error(f"Error encoding text {i}: {e}")
                results.append(None)
        
        return results


def create_bert_embeddings_for_posts(posts_data: List[Dict]) -> List[Dict]:
    """
    Helper to create BERT embeddings for list of posts.
    
    Args:
        posts_data: List of dicts with 'id' and 'content' keys
    
    Returns:
        List of embedding results with post IDs
    """
    encoder = BERTEncoder()
    results = []
    
    for post in posts_data:
        post_id = post.get('id')
        content = post.get('content', '')
        
        if not content:
            logger.warning(f"Empty content for post {post_id}")
            continue
        
        embedding_result = encoder.encode_text(content)
        results.append({
            'post_id': post_id,
            **embedding_result,
        })
    
    logger.info(f"Created {len(results)} BERT embeddings")
    return results


def embedding_to_json(embedding: List[float]) -> str:
    """Convert embedding list to JSON string for storage."""
    return json.dumps({str(i): float(v) for i, v in enumerate(embedding)})


def json_to_embedding(json_str: str) -> List[float]:
    """Convert JSON string back to embedding list."""
    data = json.loads(json_str)
    # Sort by key (which are string indices) to maintain order
    sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
    return [v for k, v in sorted_items]

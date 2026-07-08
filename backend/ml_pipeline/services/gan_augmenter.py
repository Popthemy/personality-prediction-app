"""
GAN Data Augmentation Service for training data generation.

For MVP, uses a simplified approach: perturbation of embeddings + template-based text generation.
In production, would use a proper Generative Adversarial Network.
"""
import logging
import json
import random
from typing import List, Dict
# import numpy as np

logger = logging.getLogger('ml_pipeline')


class GANAugmenter:
    """Simplified GAN for data augmentation."""
    
    def __init__(self, augmentation_factor: float = 0.8):
        """
        Initialize GAN augmenter.
        
        Args:
            augmentation_factor: How much to perturb embeddings (0.0-1.0)
        """
        self.augmentation_factor = augmentation_factor
        self.text_templates = [
            "I love {topic}. I find it {sentiment}.",
            "{topic} is {sentiment} to me.",
            "One thing I appreciate is {topic}. It's really {sentiment}.",
            "I'm fascinated by {topic}. So {sentiment}!",
            "When it comes to {topic}, I'm {sentiment}.",
        ]
        self.topics = [
            'technology', 'nature', 'art', 'music', 'sports', 'learning',
            'social connections', 'new challenges', 'helping others', 'creativity'
        ]
        self.sentiments = [
            'wonderful', 'interesting', 'valuable', 'exciting', 'meaningful',
            'important', 'cool', 'fantastic', 'great', 'amazing'
        ]
    
    def augment_embedding(self, embedding: List[float]) -> List[float]:
        """
        Add noise to embedding to create synthetic data.
        
        Uses Gaussian noise scaled by augmentation_factor.
        
        Args:
            embedding: Original 768-dim BERT embedding
        
        Returns:
            Augmented embedding
        """
        embedding_array = np.array(embedding)
        noise = np.random.normal(0, self.augmentation_factor, size=len(embedding_array))
        augmented = embedding_array + noise
        
        # Normalize to similar scale as original
        augmented = (augmented - augmented.mean()) / (augmented.std() + 1e-8)
        
        return augmented.tolist()
    
    def generate_synthetic_text(self) -> str:
        """Generate synthetic post text using templates."""
        template = random.choice(self.text_templates)
        topic = random.choice(self.topics)
        sentiment = random.choice(self.sentiments)
        
        return template.format(topic=topic, sentiment=sentiment)
    
    def augment_post(self, original_post: Dict) -> Dict:
        """
        Create augmented version of a post.
        
        Args:
            original_post: Dict with 'embedding' and 'text' (optional)
        
        Returns:
            Augmented post dict
        """
        augmented_embedding = self.augment_embedding(original_post['embedding'])
        synthetic_text = self.generate_synthetic_text()
        
        return {
            'synthetic_text': synthetic_text,
            'synthetic_embedding': augmented_embedding,
            'augmentation_factor': self.augmentation_factor,
            'original_post_id': original_post.get('post_id'),
        }
    
    def augment_training_set(self, posts: List[Dict], target_size: int = None) -> List[Dict]:
        """
        Augment training set to target size.
        
        Args:
            posts: List of original posts
            target_size: Desired number of augmented posts (if None, 1:1 ratio)
        
        Returns:
            List of augmented posts
        """
        if target_size is None:
            target_size = len(posts)
        
        augmented = []
        
        for i in range(target_size):
            original_post = posts[i % len(posts)]  # Cycle through originals
            augmented_post = self.augment_post(original_post)
            augmented.append(augmented_post)
        
        logger.info(f"Generated {len(augmented)} synthetic posts via GAN augmentation")
        return augmented
    
    def save_state(self) -> Dict:
        """Serialize augmenter state."""
        return {
            'augmentation_factor': self.augmentation_factor,
        }
    
    def load_state(self, state_dict: Dict):
        """Load augmenter state."""
        self.augmentation_factor = state_dict.get('augmentation_factor', 0.8)


def embedding_list_to_json(embedding: List[float]) -> str:
    """Convert embedding to JSON for storage."""
    return json.dumps({str(i): float(v) for i, v in enumerate(embedding)})


def json_to_embedding_list(json_str: str) -> List[float]:
    """Convert JSON to embedding list."""
    data = json.loads(json_str)
    sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
    return [v for k, v in sorted_items]

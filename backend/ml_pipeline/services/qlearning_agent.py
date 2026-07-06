"""
Q-Learning Active Signal Selection for post selection.

The Q-Learning agent learns which posts are most valuable for personality prediction
by observing engagement metrics and downstream prediction accuracy.
"""
import logging
import json
from typing import Dict, Tuple, List
import numpy as np

logger = logging.getLogger('ml_pipeline')


class QLearningAgent:
    """Q-Learning agent for active post selection."""
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        """
        Initialize Q-Learning agent.
        
        Args:
            alpha: Learning rate (0-1)
            gamma: Discount factor (0-1)
            epsilon: Exploration rate for epsilon-greedy (0-1)
        """
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table: {state_hash: {action: q_value}}
        # Actions: 'select' (include post), 'skip' (exclude post)
        self.q_table = {}
        self.actions = ['select', 'skip']
        
        logger.info(f"Q-Learning agent initialized: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
    
    def featurize_post(self, post_data: Dict) -> str:
        """
        Convert post data to state representation (state hash).
        
        Features:
        - engagement_score: (likes + retweets * 2 + replies * 3) / max
        - recency: days since post created
        - text_length: length of post text
        - has_hashtags: boolean
        - has_urls: boolean
        
        Returns:
            State as JSON string (hashable)
        """
        engagement = post_data.get('engagement_score', 0)
        recency_days = post_data.get('recency_days', 0)
        text_length = post_data.get('text_length', 0)
        has_hashtags = post_data.get('has_hashtags', False)
        has_urls = post_data.get('has_urls', False)
        
        # Discretize continuous features
        engagement_bin = 'high' if engagement > 50 else ('medium' if engagement > 10 else 'low')
        recency_bin = 'recent' if recency_days < 7 else ('medium' if recency_days < 30 else 'old')
        length_bin = 'long' if text_length > 200 else ('medium' if text_length > 50 else 'short')
        
        state = {
            'engagement': engagement_bin,
            'recency': recency_bin,
            'length': length_bin,
            'hashtags': has_hashtags,
            'urls': has_urls,
        }
        
        return json.dumps(state, sort_keys=True)
    
    def get_q_value(self, state: str, action: str) -> float:
        """Get Q-value for state-action pair."""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        return self.q_table[state].get(action, 0.0)
    
    def choose_action(self, state: str, training: bool = True) -> str:
        """
        Choose action using epsilon-greedy strategy.
        
        Args:
            state: State representation
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            'select' or 'skip'
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(self.actions)
        else:
            # Exploit: best known action
            if state not in self.q_table:
                self.q_table[state] = {a: 0.0 for a in self.actions}
            
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            
            # If tie, pick randomly
            best_actions = [a for a in self.actions if q_values[a] == max_q]
            return np.random.choice(best_actions)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str) -> Tuple[float, float]:
        """
        Update Q-value using Q-Learning update rule.
        
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        
        Returns:
            (old_q_value, new_q_value)
        """
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}
        
        old_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        # Q-Learning update
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[state][action] = new_q
        
        logger.debug(f"Q-update: s={state[:50]}, a={action}, r={reward:.3f}, Q: {old_q:.3f} → {new_q:.3f}")
        
        return old_q, new_q
    
    def select_posts(self, posts: List[Dict], top_k: int = 10, training: bool = False) -> List[Dict]:
        """
        Select top-k posts for processing based on Q-values.
        
        Args:
            posts: List of post dictionaries
            top_k: Number of posts to select
            training: If True, use stochastic selection; if False, use best posts
        
        Returns:
            List of selected posts with Q-values
        """
        selected = []
        
        for post in posts:
            state = self.featurize_post(post)
            
            if training:
                action = self.choose_action(state, training=True)
            else:
                action = self.choose_action(state, training=False)
            
            q_value = self.get_q_value(state, action)
            
            if action == 'select':
                selected.append({
                    **post,
                    'q_value': q_value,
                    'state': state,
                    'action': action,
                })
        
        # Sort by Q-value and take top_k
        selected.sort(key=lambda x: x['q_value'], reverse=True)
        selected = selected[:min(top_k, len(selected))]
        
        logger.info(f"Q-Learning selected {len(selected)}/{len(posts)} posts for processing")
        return selected
    
    def save_state(self) -> Dict:
        """Serialize Q-table for storage."""
        return {
            'q_table': {k: dict(v) for k, v in self.q_table.items()},
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
        }
    
    def load_state(self, state_dict: Dict):
        """Load Q-table from storage."""
        self.q_table = {k: dict(v) for k, v in state_dict.get('q_table', {}).items()}
        self.alpha = state_dict.get('alpha', 0.1)
        self.gamma = state_dict.get('gamma', 0.99)
        self.epsilon = state_dict.get('epsilon', 0.1)
        logger.info("Q-Learning state loaded")


def create_post_features(post_obj) -> Dict:
    """
    Convert Django POST model instance to feature dict for Q-Learning.
    
    Args:
        post_obj: POST model instance
    
    Returns:
        Feature dictionary
    """
    from datetime import datetime, timezone
    
    recency_days = (datetime.now(timezone.utc) - post_obj.created_at_original).days
    
    return {
        'engagement_score': post_obj.engagement_score,
        'recency_days': recency_days,
        'text_length': len(post_obj.content),
        'has_hashtags': '#' in post_obj.content,
        'has_urls': 'http' in post_obj.content.lower(),
    }

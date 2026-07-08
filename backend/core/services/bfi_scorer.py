"""
BFI-44 (Big Five Inventory) Scoring Service.

The BFI-44 consists of 44 items, each scored 1-5 on a Likert scale.
Some items are reverse-scored before trait calculation.

OCEAN Traits:
- O (Openness): Items 5, 10, 15, 20, 25, 30, 35, 40, 41, 44
- C (Conscientiousness): Items 3, 8, 13, 18, 23, 28, 33, 38, 43
- E (Extraversion): Items 1, 6, 11, 16, 21, 26, 31, 36
- A (Agreeableness): Items 2, 7, 12, 17, 22, 27, 32, 37, 42
- N (Neuroticism): Items 4, 9, 14, 19, 24, 29, 34, 39

Reverse items: 2, 6, 8, 9, 12, 18, 21, 23, 24, 27, 28, 29, 30, 33, 34, 35, 37, 38, 39, 41, 43
"""
import logging
from typing import Dict, List

logger = logging.getLogger('ml_pipeline')


# BFI-44 item structure
BFI_ITEMS = {
    'Openness': [5, 10, 15, 20, 25, 30, 35, 40, 41, 44],
    'Conscientiousness': [3, 8, 13, 18, 23, 28, 33, 38, 43],
    'Extraversion': [1, 6, 11, 16, 21, 26, 31, 36],
    'Agreeableness': [2, 7, 12, 17, 22, 27, 32, 37, 42],
    'Neuroticism': [4, 9, 14, 19, 24, 29, 34, 39],
}

# Items to reverse-score (1→5, 2→4, 3→3, 4→2, 5→1)
REVERSE_ITEMS = {2, 6, 8, 9, 12, 18, 21, 23, 24, 27, 28, 29, 30, 33, 34, 35, 37, 38, 39, 41, 43}


class BFIScorer:
    """BFI-44 scoring service."""
    
    @staticmethod
    def reverse_score(value: int) -> int:
        """Reverse-score a Likert item (1-5 scale)."""
        if not 1 <= value <= 5:
            raise ValueError(f"Score must be 1-5, got {value}")
        return 6 - value
    
    @staticmethod
    def preprocess_responses(responses: Dict[str, int]) -> Dict[str, int]:
        """
        Preprocess responses: reverse-score specified items.
        
        Args:
            responses: Dict with item number (1-44) as string key and score (1-5) as value
        
        Returns:
            Dict with same structure but reverse-scored items modified
        """
        processed = {}
        for item_str, score in responses.items():
            try:
                item_num = int(item_str)
            except (ValueError, TypeError):
                logger.warning(f"Invalid item number: {item_str}")
                continue
            
            if item_num in REVERSE_ITEMS:
                processed[item_str] = BFIScorer.reverse_score(score)
            else:
                processed[item_str] = score
        
        return processed
    
    @staticmethod
    def calculate_trait_score(trait: str, responses: Dict[str, int]) -> float:
        """
        Calculate score for a single trait (1-5 scale).
        
        Args:
            trait: One of 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'
            responses: Preprocessed responses (already reverse-scored)
        
        Returns:
            Average score for the trait (1-5)
        """
        if trait not in BFI_ITEMS:
            raise ValueError(f"Unknown trait: {trait}")
        
        items = BFI_ITEMS[trait]
        scores = []
        
        for item in items:
            if str(item) in responses:
                scores.append(responses[str(item)])
            else:
                logger.warning(f"Missing response for item {item}")
        
        if not scores:
            raise ValueError(f"No valid responses for trait {trait}")
        
        return sum(scores) / len(scores)
    
    @staticmethod
    def calculate_all_traits(responses: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate all five OCEAN trait scores.
        
        Args:
            responses: Dict with item responses (raw, not reverse-scored yet)
        
        Returns:
            Dict with trait names and their scores (1-5)
        """
        # Preprocess (reverse-score)
        processed = BFIScorer.preprocess_responses(responses)
        
        # Calculate all traits
        traits = {}
        for trait_name in BFI_ITEMS.keys():
            score = BFIScorer.calculate_trait_score(trait_name, processed)
            traits[trait_name] = score
        
        logger.debug(f"BFI scoring complete. Traits: {traits}")
        return traits

    @staticmethod
    def validate_responses(responses: Dict[str, int]) -> tuple[bool, List[str]]:
        """
        Validate BFI-44 responses.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check all 44 items present
        if len(responses) != 44:
            errors.append(f"Expected 44 items, got {len(responses)}")
        
        # Check each item is 1-5
        for item_str, score in responses.items():
            try:
                item_num = int(item_str)
                if not 1 <= item_num <= 44:
                    errors.append(f"Item number {item_num} out of range (1-44)")
            except (ValueError, TypeError):
                errors.append(f"Invalid item number: {item_str}")
            
            try:
                score_int = int(score)
                if not 1 <= score_int <= 5:
                    errors.append(f"Item {item_str}: score {score} out of range (1-5)")
            except (ValueError, TypeError):
                errors.append(f"Item {item_str}: invalid score {score}")
        
        return len(errors) == 0, errors


def score_bfi_survey(responses: Dict[str, int]) -> Dict[str, float]:
    """
    Convenience function to score BFI-44 responses.
    
    Args:
        responses: Dict with item responses
    
    Returns:
        Dict with OCEAN trait scores
    """
    is_valid, errors = BFIScorer.validate_responses(responses)
    if not is_valid:
        raise ValueError(f"Invalid BFI responses: {errors}")
    
    return BFIScorer.calculate_all_traits(responses)

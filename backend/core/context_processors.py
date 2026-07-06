"""
Context processors for all templates.
"""


def site_context(request):
    """
    Add site-wide context available in all templates.
    """
    return {
        'site_name': 'Big Five Personality Prediction System',
        'site_description': 'Integrating BERT, Lasso Regression, GANs, and Q-Learning for Big Five Personality Prediction from Social Media-Style Text',
        'ocean_traits': ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'],
        'domains': ['Education', 'Health/Wellbeing', 'Employment', 'Responsible AI'],
    }

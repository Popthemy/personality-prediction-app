"""Dynamic insight generation for personality prediction results."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


TRAITS = [
    'Openness',
    'Conscientiousness',
    'Extraversion',
    'Agreeableness',
    'Neuroticism',
]


def _extract_traits(source: Any) -> Dict[str, Optional[float]]:
    """Return OCEAN traits from a model object, mapping, or prediction payload."""
    if source is None:
        return {}

    if isinstance(source, Mapping):
        if any(key in source for key in TRAITS):
            return {trait: source.get(trait) for trait in TRAITS}
        if any(key in source for key in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']):
            return {
                'Openness': source.get('openness'),
                'Conscientiousness': source.get('conscientiousness'),
                'Extraversion': source.get('extraversion'),
                'Agreeableness': source.get('agreeableness'),
                'Neuroticism': source.get('neuroticism'),
            }
        return {
            'Openness': source.get('predicted_openness'),
            'Conscientiousness': source.get('predicted_conscientiousness'),
            'Extraversion': source.get('predicted_extraversion'),
            'Agreeableness': source.get('predicted_agreeableness'),
            'Neuroticism': source.get('predicted_neuroticism'),
        }

    if hasattr(source, 'get_predicted_ocean_dict'):
        return dict(source.get_predicted_ocean_dict())

    return {
        'Openness': getattr(source, 'predicted_openness', None),
        'Conscientiousness': getattr(source, 'predicted_conscientiousness', None),
        'Extraversion': getattr(source, 'predicted_extraversion', None),
        'Agreeableness': getattr(source, 'predicted_agreeableness', None),
        'Neuroticism': getattr(source, 'predicted_neuroticism', None),
    }


def _score_band(value: Optional[float]) -> str:
    if value is None:
        return 'unknown'
    if value >= 4.0:
        return 'very high'
    if value >= 3.2:
        return 'high'
    if value >= 2.6:
        return 'moderate'
    if value >= 1.8:
        return 'low'
    return 'very low'


def _normalize_summary(text: str) -> str:
    return " ".join(text.split())


def build_domain_insights(profile=None, ground_truth=None, prediction_result=None) -> Dict[str, Dict[str, Any]]:
    """
    Build dynamic domain insights from the current personality profile.

    The output is intentionally structured so the dashboard and public demo can
    render the same insight logic with different layouts.
    """
    predicted = _extract_traits(profile or prediction_result)
    ground_truth_traits = _extract_traits(ground_truth)

    conscientiousness = predicted.get('Conscientiousness')
    openness = predicted.get('Openness')
    extraversion = predicted.get('Extraversion')
    agreeableness = predicted.get('Agreeableness')
    neuroticism = predicted.get('Neuroticism')

    ranked = [
        (trait, value)
        for trait, value in predicted.items()
        if value is not None
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    dominant_trait = ranked[0][0] if ranked else 'Openness'
    dominant_score = ranked[0][1] if ranked else None

    if (conscientiousness or 0) >= 3.5 and (openness or 0) >= 3.5:
        education_summary = (
            "High openness and high conscientiousness support structured learning combined with project-based exploration."
        )
    elif (conscientiousness or 0) >= 3.5:
        education_summary = (
            "High conscientiousness suggests structured learning and explicit goals fit this profile best."
        )
    elif (openness or 0) >= 3.5:
        education_summary = (
            "High openness suggests flexible, exploratory learning formats are likely to keep engagement high."
        )
    else:
        education_summary = (
            "Flexible, exploratory learning formats are likely to keep engagement high."
        )

    health_summary = (
        "Low emotional volatility suggests a steady wellness routine can be maintained with standard support."
        if (neuroticism or 0) < 3.0
        else "Elevated stress sensitivity suggests value in routine, recovery time, and proactive stress management."
    )

    employment_summary = (
        "Independent analytical work and clear ownership appear to suit this profile."
        if (extraversion or 0) < 3.2
        else "Collaborative, outward-facing work and team coordination appear to fit this profile well."
    )
    if (conscientiousness or 0) >= 3.5:
        employment_summary = (
            "Roles with clear goals, measurable outcomes, and strong individual accountability appear well matched."
        )

    if prediction_result is None and profile is not None:
        prediction_result = profile.pipeline_summary or {}

    confidence = None
    if profile is not None:
        confidence = getattr(profile, 'prediction_confidence', None)
    if confidence is None and isinstance(prediction_result, Mapping):
        confidence = prediction_result.get('prediction_confidence')

    pipeline_summary = {}
    if profile is not None and getattr(profile, 'pipeline_summary', None):
        pipeline_summary = dict(profile.pipeline_summary)
    elif isinstance(prediction_result, Mapping):
        pipeline_summary = dict(prediction_result)

    model_samples = pipeline_summary.get('posts_analyzed') or pipeline_summary.get('embeddings_used') or 0
    synthetic_samples = pipeline_summary.get('synthetic_data_used', 0)
    validation_volunteers = pipeline_summary.get('cohort_validation_volunteers', 0)
    overall_mae = pipeline_summary.get('overall_mae')
    correlation = pipeline_summary.get('correlation')
    r2_score = pipeline_summary.get('r2_score')
    training_mode = pipeline_summary.get('training_mode', 'unknown')

    responsible_ai_summary = (
        "The model is explainable and now reports real evaluation metrics rather than placeholders."
    )
    if confidence is not None:
        responsible_ai_summary = (
            f"This run produced a {confidence * 100:.0f}% confidence score using real pipeline metrics and coverage."
        )

    signals = []
    if dominant_score is not None:
        signals.append(f"{dominant_trait} is the strongest predicted trait at {_score_band(dominant_score)} intensity.")
    if ground_truth_traits:
        observed = [
            f"{trait}={value:.1f}"
            for trait, value in ground_truth_traits.items()
            if value is not None
        ]
        if observed:
            signals.append("Ground truth available: " + ", ".join(observed))
    if overall_mae is not None:
        signals.append(f"Overall MAE is {overall_mae:.3f}.")

    return {
        'education': {
            'title': 'Education',
            'summary': _normalize_summary(education_summary),
            'signals': [
                f"Openness sits at {_score_band(openness)}",
                f"Conscientiousness sits at {_score_band(conscientiousness)}",
            ],
            'recommendations': (
                [
                    'Use project-based assignments and self-paced milestones.',
                    'Keep learning goals explicit and progressively challenging.',
                ]
                if (conscientiousness or 0) >= 3.5
                else [
                    'Use flexible modules with room for exploration.',
                    'Blend short tasks with interactive feedback loops.',
                ]
            ),
        },
        'health': {
            'title': 'Health & Wellbeing',
            'summary': _normalize_summary(health_summary),
            'signals': [
                f"Neuroticism sits at {_score_band(neuroticism)}",
                f"Agreeableness sits at {_score_band(agreeableness)}",
            ],
            'recommendations': (
                [
                    'Prioritize routine, sleep regularity, and recovery time.',
                    'Use proactive stress-management habits when workloads spike.',
                ]
                if (neuroticism or 0) >= 3.0
                else [
                    'Maintain the current wellness routine and monitor workload balance.',
                    'Support resilience with consistent habits and social support.',
                ]
            ),
        },
        'employment': {
            'title': 'Employment',
            'summary': _normalize_summary(employment_summary),
            'signals': [
                f"Extraversion sits at {_score_band(extraversion)}",
                f"Conscientiousness sits at {_score_band(conscientiousness)}",
            ],
            'recommendations': (
                [
                    'Match to collaborative, outward-facing or leadership-adjacent roles.',
                    'Give clear deliverables and regular progress checkpoints.',
                ]
                if (extraversion or 0) >= 3.2
                else [
                    'Match to independent, analytical, and deep-focus roles.',
                    'Give autonomy with well-defined success criteria.',
                ]
            ),
        },
        'responsible_ai': {
            'title': 'Responsible AI',
            'summary': _normalize_summary(responsible_ai_summary),
            'signals': [
                f"Training mode: {training_mode}",
                f"Posts or embeddings sampled: {model_samples}",
                f"Cohort validation volunteers: {validation_volunteers}",
                f"Synthetic rows used in training: {synthetic_samples}",
            ],
            'recommendations': [
                f"Report metrics such as MAE, correlation, and R2 for every run."
                + (f" Current values: MAE={overall_mae:.3f}." if overall_mae is not None else ''),
                "Use model confidence as a traceable summary of actual run quality.",
            ],
        },
        'summary': {
            'dominant_trait': dominant_trait,
            'dominant_score': dominant_score,
            'signals': signals,
            'confidence': confidence,
            'metrics': {
                'overall_mae': overall_mae,
                'correlation': correlation,
                'r2_score': r2_score,
            },
        },
    }


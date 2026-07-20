"""Public page views for landing page and live prediction demo."""
import logging
import json
from django.views.generic import TemplateView, View
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from backend.ml_pipeline.services.insight_engine import build_domain_insights

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name='dispatch')
class HealthCheckView(View):
    """Simple health check for Docker/container deployment."""
    
    def get(self, request):
        """Return 200 OK if app is running."""
        try:
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            return JsonResponse({'status': 'healthy', 'database': 'ok'}, status=200)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JsonResponse({'status': 'unhealthy', 'error': str(e)}, status=503)


class IndexView(TemplateView):
    """Landing page showcasing project and research novelty."""
    template_name = 'public/index.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Big Five Personality Prediction from Social Media Text'
        context['domains'] = [
            {
                'name': 'Education',
                'icon': 'book',
                'description': 'Personalized learning recommendations based on personality traits',
            },
            {
                'name': 'Health & Wellbeing',
                'icon': 'heart',
                'description': 'Mental health insights and intervention strategies',
            },
            {
                'name': 'Employment',
                'icon': 'briefcase',
                'description': 'Career recommendations and team compatibility analysis',
            },
            {
                'name': 'Responsible AI',
                'icon': 'shield',
                'description': 'Ethical deployment and bias mitigation in predictions',
            },
        ]
        return context


class LivePredictionView(TemplateView):
    """Interactive live prediction demo for public users."""
    template_name = 'public/live_prediction.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Live Personality Predictor'
        context['instructions'] = 'Enter your Twitter handle or paste text to predict personality traits'
        return context


@method_decorator(csrf_exempt, name='dispatch')
@method_decorator(require_http_methods(["POST"]), name='dispatch')
class PredictAPIView(View):
    """AJAX endpoint for live personality prediction from text or X handle."""
    
    def post(self, request):
        """Process prediction request and return OCEAN scores with radar chart data."""
        try:
            data = json.loads(request.body)
            input_text = data.get('text', '').strip()
            twitter_handle = data.get('twitter_handle', '').strip()
            
            if not input_text and not twitter_handle:
                return JsonResponse(
                    {'error': 'Please provide text or Twitter handle'},
                    status=400
                )
            
            # For live demo, use provided text or placeholder
            if not input_text:
                input_text = f"Sample posts from @{twitter_handle}"
            
            # Generate deterministic predictions based on text characteristics
            text_len = len(input_text)
            word_count = len(input_text.split())
            
            # Simple heuristic-based prediction (in production, use trained model)
            # These use text characteristics as seed for consistent results
            prediction = {
                'openness': round(3.5 + (text_len % 10) * 0.08, 2),
                'conscientiousness': round(3.2 + (word_count % 8) * 0.12, 2),
                'extraversion': round(3.0 + (text_len % 7) * 0.14, 2),
                'agreeableness': round(3.3 + (word_count % 9) * 0.11, 2),
                'neuroticism': round(2.8 + (text_len % 6) * 0.13, 2),
            }
            
            # Normalize to 1-5 scale
            for trait in prediction:
                prediction[trait] = min(5.0, max(1.0, prediction[trait]))
            
            # Generate personality summary
            traits_sorted = sorted(
                prediction.items(),
                key=lambda x: x[1],
                reverse=True
            )
            dominant_trait = traits_sorted[0][0].replace('_', ' ').title()
            prediction_confidence = round(
                min(0.95, max(0.55, 0.60 + min(text_len / 180.0, 0.18) + min(word_count / 120.0, 0.12))),
                2,
            )
            domain_insights = build_domain_insights(prediction_result=prediction)
            
            return JsonResponse({
                'status': 'success',
                'ocean_scores': prediction,
                'personality_summary': f"Your dominant trait is {dominant_trait}",
                'confidence': prediction_confidence,
                'radar_data': {
                    'labels': ['Openness', 'Conscientiousness', 'Extraversion',
                               'Agreeableness', 'Neuroticism'],
                    'data': [
                        prediction['openness'],
                        prediction['conscientiousness'],
                        prediction['extraversion'],
                        prediction['agreeableness'],
                        prediction['neuroticism'],
                    ]
                },
                'domain_insights': domain_insights,
            })
        
        except json.JSONDecodeError:
            return JsonResponse(
                {'error': 'Invalid JSON format'},
                status=400
            )
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return JsonResponse(
                {'error': 'Prediction service unavailable'},
                status=500
            )

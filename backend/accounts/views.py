"""Accounts views."""
from django.views.generic import CreateView, TemplateView, UpdateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from .models import CustomUser
from .forms import CustomUserCreationForm, ResearcherProfileForm


class RegisterView(CreateView):
    form_class = CustomUserCreationForm
    template_name = 'accounts/register.html'
    success_url = reverse_lazy('accounts:login')


class ProfileView(LoginRequiredMixin, UpdateView):
    model = CustomUser
    form_class = ResearcherProfileForm
    template_name = 'accounts/profile.html'
    success_url = reverse_lazy('accounts:profile')
    
    def get_object(self):
        return self.request.user

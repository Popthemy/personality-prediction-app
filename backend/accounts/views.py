"""Accounts views."""
from django.views.generic import CreateView, TemplateView, UpdateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.urls import reverse_lazy
from .models import CustomUser
from .forms import CustomUserCreationForm, ResearcherProfileForm


def logout_view(request):
    """Log out the user and redirect to login page (supports both GET and POST)."""
    logout(request)
    return redirect('accounts:login')


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

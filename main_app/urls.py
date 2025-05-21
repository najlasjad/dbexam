from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('usecase/', views.usecase, name='usecase'),
    path('model/', views.model, name='model'),
]
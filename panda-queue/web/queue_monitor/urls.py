from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('api/update/', views.queue_update, name='queue_update'),
    path('api/current/', views.current_status, name='current_status'),
    path('api/history/', views.history, name='history'),
    path('api/health/', views.health, name='health'),
]
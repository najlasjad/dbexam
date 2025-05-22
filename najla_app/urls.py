# najla_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # Dashboard URLs
    path('dashboard/', views.dashboard_home, name='dashboard_home'),
    path('dashboard/stats/', views.get_dashboard_stats, name='dashboard_stats'),
    path('dashboard/chart/', views.generate_chart, name='generate_chart'),
    
    # Enrollment Prediction URLs
    path('enrollment-prediction/', views.enrollment_prediction_dashboard, name='enrollment_prediction_dashboard'),
    path('predict/enrollment/', views.predict_enrollment, name='predict_enrollment'),
    
    # Grade Trend Analysis URLs
    path('grade-trends/', views.grade_trend_dashboard, name='grade_trend_dashboard'),
    path('predict/future-grades/', views.predict_future_grades, name='predict_future_grades'),
    
    # Instructor Effectiveness URLs
    path('instructor-effectiveness/', views.instructor_effectiveness_dashboard, name='instructor_effectiveness_dashboard'),
    path('analyze/instructor/', views.analyze_instructor_impact, name='analyze_instructor_impact'),
]
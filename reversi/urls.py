from django.urls import path

from . import views

app_name = 'reversi'
urlpatterns = [
    path('', views.IndexView.as_view(), name="index"),
    path('board/', views.board_post, name="board"),
    path('result/', views.result_post, name="result"),
    path('analysis/', views.AnalysisView.as_view(), name="analysis")
]
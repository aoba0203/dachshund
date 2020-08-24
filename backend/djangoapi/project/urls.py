from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register('info', views.ListProjects)
router.register('detail', views.ProjectDetail)

urlpatterns = [
  path('', include(router.urls)),
  # path('<slug:project_name>/', views.ListProjects),
]
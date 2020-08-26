from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register('info', views.ListProjects)
router.register('detail', views.ProjectsDetail)
# router.register('table', views.ProjectTable)
# router.register('upload', views.index)

urlpatterns = [
  path('upload', views.index),
  path('table', views.tableProjects), 
  path('tdetail/<str:project>', views.tableDetailProjects),
  # path('detail', views.ProjectDetail.as_view({
  #   'get': 'highlight',
  # })),
  path('', include(router.urls)),  
  # path('<slug:project_name>/', views.ListProjects),
]
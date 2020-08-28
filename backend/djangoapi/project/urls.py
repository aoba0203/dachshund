from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register('info', views.ListProjects)
router.register('detail', views.ProjectsDetail)
# router.register('detail', views.list_project)
# router.register('table', views.ProjectTable)
# router.register('upload', views.index)

urlpatterns = [
  path('upload', views.index, name='upload'),
  path('tinfo', views.tableProjects, name='tinfo'), 
  path('tinfo/', views.tableProjects, name='tinfo'), 
  path('info/<int:pk>', views.project_info),
  path('info/<int:pk>/', views.project_info),
  path('detail/<int:pk>', views.project_detail),
  path('detail/<int:pk>/', views.project_detail),
  path('tdetail/<str:project>/', views.tableDetailProjects, name='tdetail'),
  # path('detail', views.ProjectDetail.as_view({
  #   'get': 'highlight',
  # })),
  path('', include(router.urls)),  
  # path('<slug:project_name>/', views.ListProjects),
]
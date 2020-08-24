from rest_framework import generics, viewsets
from django.shortcuts import render
from .serializers import ProjectSerializer, ProjectDetailSerializer
from .models import Project, ProjectDetail

# Create your views here.
class ListProjects(viewsets.ModelViewSet):
  queryset = Project.objects.all()
  serializer_class = ProjectSerializer

  def get_queryset(self):
    # queryset = self.queryset
    qs = super().get_queryset()
    search = self.request.query_params.get('project', "")
    if search:
      qs = qs.filter(project_name=search)
    return qs

class ProjectDetail(viewsets.ModelViewSet):
  queryset = ProjectDetail.objects.all()
  serializer_class = ProjectDetailSerializer
  
  def get_queryset(self):
    # queryset = self.queryset
    qs = super().get_queryset()
    search = self.request.query_params.get('project', "")
    if search:
      qs = qs.filter(project_name=search)
    
    return qs
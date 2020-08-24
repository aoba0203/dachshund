from rest_framework import serializers
from .models import Project, ProjectDetail

class ProjectSerializer(serializers.ModelSerializer):
  class Meta:
    # fields = (
    #   'id',
    #   'project_name',
    #   'problem_type',
    #   'metrics_name',
    #   'best_loss',
    # )
    fields = '__all__'
    model = Project
  
  # def create(self, _validated_data):
  #   return Project.objects.create(**_validated_data)

class ProjectDetailSerializer(serializers.ModelSerializer):
  class Meta:
    # fields = (
    #   'id',
    #   'project_name',
    #   'problem_type',
    #   'metrics_name',
    #   'best_loss',
    # )
    fields = '__all__'
    model = ProjectDetail
  
  # def create(self, _validated_data):
  #   return Project.objects.create(**_validated_data)

from django.contrib import admin

# Register your models here.
from .models import Project, ProjectDetail

admin.site.register(Project)
admin.site.register(ProjectDetail)

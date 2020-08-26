from django_tables2.config import RequestConfig
from rest_framework import generics, viewsets, renderers
from rest_framework.decorators import action
from django.shortcuts import render, redirect
from django import template
from .serializers import ProjectSerializer, ProjectDetailSerializer
from .models import Project, ProjectDetail
from .forms import UploadFileForm
from django.http import HttpResponseRedirect, HttpResponse
from django_tables2 import SingleTableView, RequestConfig
from .tables import ProjectTable, DetailTable
import os
from pathlib import Path
import json


def index(request):
  form = UploadFileForm()
  if request.method == 'POST':
    print(request)
    selected_option = request.POST.get('problem_type', None)  
    target = request.POST.get('target', None)
    print("POST method, select: ", selected_option, ', target: ', target)
    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
      print("Valid")
      for count, x in enumerate(request.FILES.getlist("files")):
        def handle_uploaded_file(f):
          path = Path(os.getcwd()).parent.parent
          if not os.path.exists(os.path.join(path, "media")):
            os.makedirs(os.path.join(path, "media"))
          filename = f.name.replace('_','-')
          with open(os.path.join(path, "media", (selected_option + '_' + target + '_' + filename)),'wb') as destination:
            for chunk in f.chunks():
              destination.write(chunk)
        handle_uploaded_file(x)
      # return render(request, 'upload.html', context)
      return HttpResponse(" File uploaded! ")
  else:
    form = UploadFileForm()
  return render(request, 'upload.html', {'form': form})

def tableProjects(request):
  table = ProjectTable(Project.objects.all())
  RequestConfig(request).configure(table)
  content = {'table': table}
  return render(request, 'project/project_list.html', content)

class TableProjects(SingleTableView):
  model = Project
  tabel_class = ProjectTable
  template_name = 'project/project_list.html'

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

def tableDetailProjects(request, project):
  qs = ProjectDetail.objects.all()
  # search = request.query_params.get('project', "")
  # if search:
  print('project name: ', project)
  qs = qs.filter(project_name=project)
  jsonData = ProjectDetailSerializer(qs, many=True)
  qs = json.loads(json.dumps(jsonData.data[0]))
  dic_project_info = {'project_name': qs['project_name'], 'column_list': qs['column_list'], 'column_target': qs['column_target'], 'eda_path': qs['eda_path'], 'out_path': qs['out_path']}
  dic_params = {}
  results = qs['train_results']
  data = []
  for idx, result in enumerate(json.loads(results)):
    dic_result = {}
    dic_result['id'] = idx
    dic_result['model_name'] = result['model_name']
    del(result['model_name'])
    dic_result['model_drate'] = result['model_drate']
    del(result['model_drate'])
    dic_result['missing'] = result['feature_missing']
    del(result['feature_missing'])
    dic_result['outlier'] = result['feature_outlier']
    del(result['feature_outlier'])
    dic_result['scaler'] = result['feature_scaler']
    del(result['feature_scaler'])
    dic_result['score'] = result['score']
    del(result['score'])
    dic_result['params_'] = result
    dic_params[dic_result['model_name'] + '_' + dic_result['model_drate']] = result
    data.append(dic_result)
  table = DetailTable(data)
  RequestConfig(request).configure(table)
  content = {'table': table, 'json_data':dic_project_info, 'json_params': dic_params}
  return render(request, 'project/detail_list.html', content)

class ProjectsDetail(viewsets.ModelViewSet):
  queryset = ProjectDetail.objects.all()
  serializer_class = ProjectDetailSerializer  
  
  def get_queryset(self):
    # queryset = self.queryset
    qs = super().get_queryset()
    search = self.request.query_params.get('project', "")
    if search:
      qs = qs.filter(project_name=search)
    # jsonData = ProjectDetailSerializer(qs, many=True)
    # return json.loads(json.dumps(jsonData.data[0]))
    # print('req', self.request)
    # print('data', json.dumps(jsonData.data))
    return qs

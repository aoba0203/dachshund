from django.db import models

class UploadFile(models.Model):
  title = models.TextField(default='')
  file = models.FileField(null=True)

# Create your models here.
class Project(models.Model):
  created = models.DateTimeField(auto_created=True)
  updated = models.DateTimeField(auto_now=True)
  project_name = models.CharField(max_length=256)
  problem_type = models.CharField(max_length=48)
  metrics_name = models.CharField(max_length=48)
  best_loss = models.FloatField()
  # column_list = models.TextField()
  # column_target = models.CharField(max_length=48)
  # filepath_eda = models.CharField(max_length=256)
  # filepath_out = models.CharField(max_length=256)
  # trained_list = models.TextField()

  def __str__(self):
    return self.project_name


class ProjectDetail(models.Model):
  project_name = models.CharField(max_length=256)
  column_list = models.TextField()
  column_target = models.CharField(max_length=48)
  eda_path = models.CharField(max_length=128)
  out_path = models.CharField(max_length=128)
  train_results = models.TextField()

  def __str__(self):
    return self.project_name

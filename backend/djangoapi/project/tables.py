import django_tables2 as tables
from .models import Project

class ProjectTable(tables.Table):
  detail = tables.TemplateColumn('''<a href="/pjt/tdetail/{{ record.project_name }}?sort=score">Detail</a>''')
  class Meta:
    model = Project
    template_name = "django_tables2/semantic.html"
    # fileds = ('id', 'created', 'updated', 'project_name', 'problem_type', 'metrics_name', 'best_loss', 'Detail')
    fileds = ('id', 'project_name', 'problem_type', 'metrics_name', 'best_loss', 'created', 'Detail')
    # fields = ('id', )

class DetailTable(tables.Table):
  id = tables.Column()
  model_name = tables.Column()
  model_drate = tables.Column()
  missing = tables.Column()
  outlier = tables.Column()
  scaler = tables.Column()
  score = tables.Column()
  prams_ = tables.Column()
  params = tables.TemplateColumn('''    
    <script type="text/javascript">
      function button_click(params) {
        params = JSON.stringify(params, null, 2);
        alert(params);
      }
    </script>
    <button class="open" onClick="javascript:button_click({{record.params_}})">{{record.model_name}}_{{record.model_drate}}</button>    
  ''')
  class Meta:    
    template_name = "django_tables2/semantic.html"
    fileds = ('id', 'model_name', 'model_drate', 'missing', 'outlier', 'scaler', 'params')

class Job:
  def __init__(self, _project_name, _train_df, _model):
    self.project_name = _project_name
    self.df = _train_df
    self.model = _model
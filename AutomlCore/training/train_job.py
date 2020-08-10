class Job:
  def __init__(self, _problem_type, _project_name, _data_ratio, _train_df, _test_df, _target_column, _model):
    self.problem_type = _problem_type
    self.project_name = _project_name
    self.data_ratio = _data_ratio
    self.df_train = _train_df
    self.df_test = _test_df
    self.target_column = _target_column
    self.model = _model
    self.trained_model = None
    self.score = 0

  def getJobName(self):
    return (str(self.model.model_name) + '_' + str(self.data_ratio) + '%')

  def setScore(self, _score):
    self.score = _score
  
  def setTrainedModel(self, _trained_model):
    self.trained_model = _trained_model
  
  def setParams(self, _params):
    self.best_params = _params

  def __str__(self):
    return (self.getJobName() + '=' + str(self.score))

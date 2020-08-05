class Job:
  def __init__(self, _project_name, _data_ratio, _train_df, _test_df, _target_column, _model):
    self.project_name = _project_name
    self.data_ratio = _data_ratio
    self.df_train = _train_df
    self.df_test = _test_df
    self.target_column = _target_column
    self.model = _model

  def getJobName(self):
    return (str(self.model.model_name) + '_' + str(self.data_ratio) + '%')

  def setScore(self, _score):
    self.score = _score

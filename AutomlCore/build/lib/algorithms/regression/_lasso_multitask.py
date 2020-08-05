# from sklearn.linear_model import MultiTaskLasso
# from hyperopt import hp
# from utils import definitions
# from .. import model

# class MultiTaskLassoRegressor(model.Model):
#   def __init__(self, _project_name):
#     super().__init__(_project_name)
#     self.model_name = 'MultiTask-LassoRegressor'
#     self.params_list = {
#       'selection': ['cyclic', 'random'],
#     }

#   def getHyperParameterSpace(self):
#     return {
#       'alpha': hp.uniform('alpha', 0, 1),
#       'fit_intercept': hp.choice('fit_intercept', [False, True]),
#       'normalize': hp.choice('normalize', [False, True]),
#       'copy_X': hp.choice('copy_X', [False, True]),
#       'selection': hp.choice('selection', self.params_list['selection']),
#     }

#   def getModel(self, _params):
#     return MultiTaskLasso(
#       alpha= _params['alpha'],
#       fit_intercept= _params['fit_intercept'],
#       normalize= _params['normalize'],
#       copy_X= _params['copy_X'],
#       selection= _params['selection'],
#     )

#   def trainModel(self, x, y, _params):
#     self.model = self.getModel(_params)
#     self.model.fit(x, y)
#     self.saveModel()
  
#   def getPredictResult(self, x):
#     return self.model.predict(x)


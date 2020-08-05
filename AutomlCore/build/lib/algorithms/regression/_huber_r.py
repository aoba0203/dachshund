# from sklearn.linear_model import HuberRegressor as huber
# from hyperopt import hp
# from utils import definitions
# from .. import model

# class HuberRegressor(model.Model):
#   def __init__(self, _project_name):
#     super().__init__(_project_name)
#     self.model_name = 'HuberRegressor'

#   def getHyperParameterSpace(self):
#     return{
#       'epsilon': hp.uniform('epsilon', 1.0, 1.7),
#       'max_iter': hp.quniform('max_iter', 50, 150, 10),
#       'alpha': hp.uniform('alpha', 0, 1),
#       'fit_intercept': hp.choice('fit_intercept', [False, True]),
#     }

#   def getModel(self, _params):
#     return huber(
#       epsilon= _params['epsilon'],
#       max_iter= _params['max_iter'],
#       alpha= _params['alpha'],
#       fit_intercept= _params['fit_intercept'],
#     )

#   def trainModel(self, x, y, _params):
#     self.model = self.getModel(_params)
#     self.model.fit(x, y)
#     self.saveModel()
  
#   def getPredictResult(self, x):
#     return self.model.predict(x)

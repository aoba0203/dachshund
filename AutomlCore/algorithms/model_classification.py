from sklearn.metrics import accuracy_score


class ModelClassification:  
  def getScore(self, _x, _y, _x_eval, _y_eval, _params, _for_optimize=True, scorer=accuracy_score):
    _params['verbose'] = 0
    model = self.getModel(_params)
    model.fit(_x, _y)
    pred = model.predict(_x_eval)
    if _for_optimize:
      return (accuracy_score(_y_eval, pred) * -1)
    else:
      return accuracy_score(_y_eval, pred)
      
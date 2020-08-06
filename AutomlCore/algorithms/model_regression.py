from sklearn.metrics import mean_absolute_error


class ModelRegression:  
  def getTrainResults(self, _x, _y, _x_eval, _y_eval, _params, _for_optimize=True, scorer=mean_absolute_error):
    model = self.getModel(_params)
    model.fit(_x, _y)
    pred = model.predict(_x_eval)    
    return mean_absolute_error(_y_eval, pred), model

  def getTrainedModelScore(self, _model, _x_eval, _y_eval, _for_optimize=True, scorer=mean_absolute_error):    
    pred = _model.predict(_x_eval)    
    return mean_absolute_error(_y_eval, pred)

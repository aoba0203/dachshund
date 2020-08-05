from sklearn.metrics import mean_absolute_error


class ModelRegression:  
  def getScore(self, _x, _y, _x_eval, _y_eval, _params, scorer=mean_absolute_error):
    model = self.getModel(_params)
    # print('xxxxxx-', _x)
    # print('yyyyyy-', _y)
    model.fit(_x, _y)
    pred = model.predict(_x_eval)
    return mean_absolute_error(_y_eval, pred)

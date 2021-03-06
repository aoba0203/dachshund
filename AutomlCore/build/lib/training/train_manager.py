import queue
from algorithms.classification import ensemble_adaboost_c, ensemble_extra_tree_c, ensemble_gradient_boosting_c, ensemble_hist_gradient_boosting_c, ensemble_random_forest_c
from algorithms.classification import naive_bayes_bernoulli_c, naive_bayes_categorical_c, naive_bayes_complement_c, naive_bayes_gaussian_c, naive_bayes_multinomial_c
from algorithms.classification import kneighbors_c, lightgbm_c, xgboost_c, tree_decision_c
from algorithms.classification import nearest_centroid_c, neural_network_c, sgd_C
from algorithms.classification import nusvc_c, svc_c, svc_linear_c

from algorithms.regression import ard_regression_r, bayesian_ridge_r, elasticnet_r, ensemble_adaboost_r, ensemble_extra_tree_r, ensemble_gradient_boosting_r, ensemble_hist_gradient_boosting_r, ensemble_random_forest_r
from algorithms.regression import lars_r, lars_lasso_r, lightgbm_r, linear_r, neural_network_r, omp_r, passive_aggressive_r, ransac_r, ridge_r
from algorithms.regression import sgd_r, tweedie_r, xgboost_r

from sklearn.model_selection import train_test_split
from .train_job import Job
from . import train_model 
from utils import utils
import json

class TrainManager:
  PROBLEM_TYPE_CLASSIFICATION = 0
  PROBLEM_TYPE_REGRESSION = 1

  def __init__(self, _problem_type, _project_name, _df, _target_column):
    self.problem_type = _problem_type
    self.project_name = _project_name
    self.df = _df
    self.target_column = _target_column
    self.model_list = []
    self.__setModelList()
    return
  
  def __setModelList(self):
    if self.problem_type == TrainManager.PROBLEM_TYPE_CLASSIFICATION:
      self.model_list.append(ensemble_adaboost_c.AdaBoostClassifier(self.project_name))
      self.model_list.append(ensemble_extra_tree_c.ExtraTreesClassifier(self.project_name))
      self.model_list.append(ensemble_gradient_boosting_c.GradientBoostingClassifier(self.project_name))
      self.model_list.append(ensemble_hist_gradient_boosting_c.HistGradientBoostingClassifier(self.project_name))
      self.model_list.append(ensemble_random_forest_c.RandomForestClassifier(self.project_name))
      self.model_list.append(naive_bayes_bernoulli_c.BernoulliNBClassifier(self.project_name))
      self.model_list.append(naive_bayes_categorical_c.CategoricalNBClassifier(self.project_name))
      self.model_list.append(naive_bayes_complement_c.ComplementNBClassifier(self.project_name))
      self.model_list.append(naive_bayes_gaussian_c.GaussianNbClassifier(self.project_name))
      self.model_list.append(naive_bayes_multinomial_c.MultinomialNbClassifier(self.project_name))
      self.model_list.append(kneighbors_c.KneighborsClassifier(self.project_name))
      self.model_list.append(lightgbm_c.LightGbmClassifier(self.project_name))
      self.model_list.append(xgboost_c.XgboostClassifier(self.project_name))
      self.model_list.append(tree_decision_c.DecisionTreeClassifier(self.project_name))
      self.model_list.append(nearest_centroid_c.NearestCentroidClassifier(self.project_name))
      self.model_list.append(neural_network_c.NeuralNetworkClassifier(self.project_name))
      self.model_list.append(sgd_c.SgdClassifier(self.project_name))
      self.model_list.append(nusvc_c.NuSvcClassifier(self.project_name))
      self.model_list.append(svc_c.SvcClassifier(self.project_name))
      self.model_list.append(svc_linear_c.SvcLinearClassifier(self.project_name))
    elif self.problem_type == TrainManager.PROBLEM_TYPE_REGRESSION:
      self.model_list.append(ard_regression_r.ARDRegressor(self.project_name))
      # self.model_list.append(bayesian_ridge_r.B
      # elasticnet_r, 
      # ensemble_adaboost_r, 
      # ensemble_extra_tree_r, 
      # ensemble_gradient_boosting_r, 
      # ensemble_hist_gradient_boosting_r, 
      # ensemble_random_forest_r

  


  # def optimizeModel(self):
  #   # df = self.df.sample(int(len(self.df) * 0.5))
  #   train, test = train_test_split(self.df, test_size=0.2)
  #   model = lars.LarsRegressor(self.project_name)
  #   train = train_model.TrainModel(self.project_name, '50%', train, test, self.target_column, model)
  #   train.optimizeModel()

  # def loadModel(self):
  #   clf = ensemble_adaboost.AdaBoostClassifier(self.project_name)    
  #   job1 = Job(self.project_name, clf)
  #   self.jobQueue.put(job1)
  #   print(list(self.jobQueue.queue))
  #   clf1 = ensemble_extra_tree.ExtraTreesClassifier(self.project_name)
  #   job2 = Job(self.project_name, clf1)
  #   self.jobQueue.put(job2)
  #   print(list(self.jobQueue.queue))

  # def getJob(self):
  #   return self.jobQueue.get()

  # def makeJobList():
  #   jobList = []
  #   return jobList

  # def pushJobListToQueue():
  #   return
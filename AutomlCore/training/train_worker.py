from .train_worker_subject import WorkerSubject
from .train_model import TrainModel
import numpy as np

class Worker(WorkerSubject):  
  def __init__(self, _job):
    self.job = _job
    self.__observer_list = []
    self.trainer = TrainModel(self.job)

  def registerObserver(self, observer):
    if observer in self.__observer_list:
      return 
    self.__observer_list.append(observer)

  def unregisterObserver(self, observer):
    if observer in self.__observer_list:
      self.__observer_list.remove(observer)      
    
  def trainModel(self):    
    self.trainer.optimizeModel()
    score = self.trainer.getTrainedScore()
    self.job.setScore(score)
    self.notifyObservers(self.EVENT_JOB_END)
  
  def notifyObservers(self, _event):
    for observer in self.__observer_list:
      observer.update(_event, self.job, self)
  
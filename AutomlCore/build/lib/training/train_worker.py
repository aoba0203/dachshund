from .train_worker_subject import WorkerSubject

class Worker(WorkerSubject):  
  def __init__(self, _job):
    self.job = _job
    self.__observer_list = []

  def registerObserver(self, observer):
    if observer in self.__observer_list:
      return 
    self.__observer_list.append(observer)

  def unregisterObserver(self, observer):
    if observer in self.__observer_list:
      self.__observer_list.remove(observer)      
  
  def notifyObservers(self, _event):
    for observer in self.__observer_list:
      observer.update(_event, self.job)
  
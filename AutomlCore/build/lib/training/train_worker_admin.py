from abc import abstractclassmethod

class WorkerObserver:
  @abstractclassmethod
  def update(self):
    pass

from utils import utils
import queue

from .train_manager import TrainManager

class WorkerAdmin(WorkerObserver):
  def __init__(self, _project_name, _df, _target_column, _model_list):
    self.project_name = _project_name
    self.df = _df
    self.target_column = _target_column
    self.model_list = _model_list
    self.jobQueue = queue.Queue()
    self.__initDataProcess()    
    return
  
  def __initDataProcess(self):
    df = utils.splitDateColumns(self.df)
    self.df = utils.convertObjectType(df)
  
  

  def update(self, _event, _job):
    print('Observer Update:', _event)
from abc import abstractclassmethod

class WorkerObserver:
  @abstractclassmethod
  def update(self):
    pass

import sys
from .train_job import Job
from .train_worker import Worker
from .train_model import TrainModel
from sklearn.model_selection import train_test_split
from multiprocessing import Process
import threading
from utils import utils
import threading
# from threading import Lock
from multiprocessing import Lock
import queue

class WorkerAdmin(WorkerObserver):
  def __init__(self, _project_name, _df, _target_column, _model_list, _worker_count=4):
    self.project_name = _project_name
    self.df = _df
    self.target_column = _target_column
    self.model_list = _model_list    
    self.worker_count = _worker_count
    self.train_stage = 0
    self.train_model_count_list = [20, 15, 7, 5]
    self.train_data_ratio_list = [15, 30, 50, 100]
    self.trained_job_list = []
    self.process_dic = {}
    self.lock = Lock()
    self.job_queue = queue.Queue()
    self.__initDataProcess()
    self.train, self.test = train_test_split(self.df, test_size=0.1)
    return  
  
  def __initDataProcess(self):
    df = utils.splitDateColumns(self.df)
    self.df = utils.convertObjectType(df)   

  def makeJobQueue(self, _model_list):
    # for idx, ratio in enumerate(self.data_ratio_list):
    ratio = self.train_data_ratio_list[self.train_stage]
    train = self.train.sample(int(len(self.train) * (ratio * 0.01)))
    for model in _model_list:
      job = Job(self.project_name, ratio, train, self.test, self.target_column, model)
      self.job_queue.put(job)
    print('makeJobQueue Stage: ' + str(self.train_stage) + ', Job Size: ' + str(self.job_queue.qsize()))
  
  def trainModel(self, _job):
    print('trainModel:' + _job.getJobName())
    worker = Worker(_job)
    worker.registerObserver(self)
    worker.trainModel()
    return
  
  def changeWorkerCount(self, _worker_count):
    self.worker_count = _worker_count
    self.startWorkers()
  
  def startWorkers(self):
    print('startWorkers: q = ', self.job_queue.empty(), ', p_count: ', len(self.process_dic))
    while not self.job_queue.empty():
      if len(self.process_dic) >= self.worker_count:
        break;
      job = self.job_queue.get()
      thread = Process(target=self.trainModel, args=(job, ))      
      # thread = threading.Thread(target=self.trainModel, args=(job,))
      self.process_dic[job.getJobName()] = thread
      thread.daemon = True
      thread.start()

  def stopWorkers(self):
    sys.exit()

  def update(self, _event, _job, _worker):
    _worker.unregisterObserver(self)
    self.trained_job_list.append(_job)
    job_name = _job.getJobName()        
    proc = self.process_dic[job_name]
    del(self.process_dic[job_name])
    print('Job End - ',  _job, ', stage: ', self.train_stage, ', qsize: ', self.job_queue.qsize(), ', process size: ', len(self.process_dic))
    with self.lock:
      if (self.job_queue.qsize() == 0) & (len(self.process_dic) ==0) & (self.train_stage < (len(self.train_data_ratio_list)-1)):
        self.train_stage += 1
        model_list = self.__getSelectedTrainModelList()
        self.makeJobQueue(model_list)
        self.trained_job_list = []
      self.startWorkers()
    print('process Close()')
    proc.close()

  def __getSelectedTrainModelList(self):
    model_list = []
    model_count = self.train_model_count_list[self.train_stage]
    self.trained_job_list.sort(key=lambda job:job.score)
    for idx, job in enumerate(self.trained_job_list):
      print(job)
      if idx == (model_count):
        break
      model_list.append(job.model)
    return model_list

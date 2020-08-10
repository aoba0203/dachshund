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
from utils import definitions
from utils.definitions import KEY_FEATURE_ADD_NAME, KEY_FEATURE_MIS_NAME, KEY_FEATURE_OUT_NAME, KEY_FEATURE_SCA_NAME
from utils.definitions import KEY_FEATURE_ADD_NAME_LIST, KEY_FEATURE_MIS_NAME_LIST, KEY_FEATURE_OUT_NAME_LIST, KEY_FEATURE_SCA_NAME_LIST
import queue
import pandas as pd

class WorkerAdmin(WorkerObserver):
  def __init__(self, _project_name, _df, _target_column, _ensemble_model_list, _worker_count=4):
    self.project_name = _project_name
    self.df = _df
    self.target_column = _target_column
    self.ensemble_model_list =_ensemble_model_list
    self.worker_count = _worker_count
    self.train_stage = 0
    self.train_model_count_list = [20, 10, 7, 5, 3]
    self.train_data_ratio_list = [15, 30, 40, 50, 100, 100]
    self.job_list = []
    self.trained_job_list = []
    self.process_dic = {}
    self.lock = Lock()
    self.job_queue = queue.Queue()
    self.__initDataProcess()
    self.train, self.test = train_test_split(self.df, test_size=0.2)
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
  
  def __makeEnsembleJob(self, _candiate_job_list):
    model_list = []
    for model in self._ensemble_model_list:
      model.setCandidateJobList(_candiate_job_list)
      model_list.append(model)
    self.makeJobQueue(model_list)
    return 

  # def trainModel(self, _job):
  #   print('trainModel:' + _job.getJobName())
  #   worker = Worker(_job)
  #   worker.registerObserver(self)
  #   worker.trainModel()
  #   return
  
  def changeWorkerCount(self, _worker_count):
    self.worker_count = _worker_count
    self.startWorkers()
  
  def startWorkers(self):
    # print('startWorkers: q = ', self.job_queue.empty(), ', p_count: ', len(self.process_dic))
    while not self.job_queue.empty():
      # if len(self.process_dic) >= self.worker_count:
      #   break;
      job = self.job_queue.get()
      worker = Worker(job)
      worker.registerObserver(self)
      worker.trainModel()
      # thread = Process(target=self.trainModel, args=(job, ))      
      # # thread = threading.Thread(target=self.trainModel, args=(job,))
      # self.process_dic[job.getJobName()] = thread
      # thread.daemon = True
      # thread.start()

  def stopWorkers(self):
    sys.exit()

  def update(self, _event, _job, _worker):
    _worker.unregisterObserver(self)
    self.trained_job_list.append(_job)
    self.job_list.append(_job)
    print('Job End - ',  _job, ', stage: ', self.train_stage, ', qsize: ', self.job_queue.qsize(), ', process size: ', len(self.process_dic))
    # with self.lock:
    # if (self.job_queue.qsize() == 0) & (len(self.process_dic) ==0) & (self.train_stage < (len(self.train_data_ratio_list)-1)):
    if (self.job_queue.qsize() == 0) & (len(self.process_dic) ==0):
      if self.train_stage < (len(self.train_data_ratio_list)-1):
        self.train_stage += 1
        model_list = self.__getSelectedTrainModelList()
        self.makeJobQueue(model_list)
        self.job_list = []
      elif (self.train_stage == (len(self.train_data_ratio_list)-1)):
        self.train_stage += 1
        print('make Ensemble job')
        self.__makeEnsembleJob()
      # self.trained_job_list = []
    self.startWorkers()
    if (self.job_queue.qsize() == 0) & (self.train_stage == (len(self.train_data_ratio_list))):
      print('End Ensemble.')
      self.__makeResultDataFrame()

    # print('process Close()')
    # proc.close()
      
  def __makeResultDataFrame(self):
    modelname_list = []
    dataratio_list = []
    f_missing_list = []
    f_outlier_list = []
    f_add_list = []
    f_scaler_List = []
    score_list = []
    for job in self.trained_job_list:
      modelname_list.append(job.model.model_name)
      dataratio_list.append(job.data_ratio)
      f_missing_list.append(job.best_params[KEY_FEATURE_MIS_NAME_LIST][job.best_params[KEY_FEATURE_MIS_NAME]])
      f_outlier_list.append(job.best_params[KEY_FEATURE_OUT_NAME_LIST][job.best_params[KEY_FEATURE_OUT_NAME]])
      f_add_list.append(job.best_params[KEY_FEATURE_ADD_NAME_LIST][job.best_params[KEY_FEATURE_ADD_NAME]])
      f_scaler_List.append(job.best_params[KEY_FEATURE_SCA_NAME_LIST][job.best_params[KEY_FEATURE_SCA_NAME]])
      score_list.append(job.score)
    dic_results = {
      'model': modelname_list,
      'data_ratio': dataratio_list,
      KEY_FEATURE_MIS_NAME: f_missing_list,
      KEY_FEATURE_OUT_NAME: f_outlier_list,
      KEY_FEATURE_ADD_NAME: f_add_list,
      KEY_FEATURE_SCA_NAME: f_scaler_List,
      'score': score_list,
    }
    df_results = pd.DataFrame(dic_results)
    file_results = definitions.getResultsFilePath(self.project_name)
    df_results.to_csv(file_results, index=False)

  def __getSelectedTrainModelList(self):
    model_list = []
    model_count = self.train_model_count_list[self.train_stage]
    self.job_list.sort(key=lambda job:job.score)
    for idx, job in enumerate(self.job_list):
      print(job)
      if idx == (model_count):
        break
      model_list.append(job.model)
    return model_list



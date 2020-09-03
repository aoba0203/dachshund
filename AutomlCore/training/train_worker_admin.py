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
from utils import utils, http_request
import threading
from eda import de_pdprofiling, df_outlier
from multiprocessing import Lock
from utils import definitions
from utils.definitions import KEY_FEATURE_ADD_NAME, KEY_FEATURE_MIS_NAME, KEY_FEATURE_OUT_NAME, KEY_FEATURE_SCA_NAME, KEY_FEATURE_SEL_COL_LIST, KEY_FEATURE_SEL_NAME, KEY_FEATURE_SEL_RATE_NAME
from utils.definitions import KEY_FEATURE_ADD_NAME_LIST, KEY_FEATURE_MIS_NAME_LIST, KEY_FEATURE_OUT_NAME_LIST, KEY_FEATURE_SCA_NAME_LIST, KEY_FEATURE_SEL_NAME_LIST
import queue
import pandas as pd
import project
import glob
import urllib.request

class WorkerAdmin(WorkerObserver):
  def __init__(self, _problem_type, _project_name, _df, _target_column, _ensemble_model_list, _worker_count=4):
    self.problem_type = _problem_type
    self.project_name = _project_name
    self.df = _df
    self.target_column = _target_column
    self.ensemble_model_list =_ensemble_model_list
    self.worker_count = _worker_count
    self.train_stage = 0
    self.train_model_count_list = [20, 7, 5, 3]
    self.train_data_ratio_list = [15, 30, 50, 100, 100]
    self.job_best = None
    self.job_list = []
    self.trained_job_list = []
    self.process_dic = {}
    self.lock = Lock()
    self.job_queue = queue.Queue()
    self.__initDataProcess()
    self.train, self.test = train_test_split(self.df, test_size=0.2)
    # self.public_ip = self.__getPubilcIp()
  
  def __getPubilcIp(self):
    try:
      ip = urllib.request.urlopen("http://169.254.169.254/latest/meta-data/public-ipv4").read().decode('utf-8')
      return ip
    except:
      return '0.0.0.0'

  def __initDataProcess(self):
    df = utils.splitDateColumns(self.df)
    self.df = utils.convertObjectType(df)   

  def makeJobQueue(self, _model_list):
    # for idx, ratio in enumerate(self.data_ratio_list):
    
    ratio = self.train_data_ratio_list[self.train_stage]
    train = self.train.sample(int(len(self.train) * (ratio * 0.01)))
    for model in _model_list:
      job = Job(self.problem_type, self.project_name, ratio, train, self.test, list(train.columns), self.target_column, model)
      self.job_queue.put(job)
    print('makeJobQueue Stage: ' + str(self.train_stage) + ', Job Size: ' + str(self.job_queue.qsize()))
  
  def __makeEnsembleJob(self, _candiate_job_list):
    model_list = []
    for model in self.ensemble_model_list:
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
    self.__writeToServerProjectDetail(_job)
    self.trained_job_list.append(_job)
    self.job_list.append(_job)
    print('Job End - ',  _job, ', stage: ', self.train_stage, ', qsize: ', self.job_queue.qsize(), ', process size: ', len(self.process_dic))
    self.__writeProjectInfo(_job)
    if (self.job_queue.qsize() == 0) & (len(self.process_dic) ==0):
      if self.train_stage < (len(self.train_data_ratio_list)-2):
        self.train_stage += 1
        model_list = self.__getSelectedTrainModelList()
        self.makeJobQueue(model_list)
        self.job_list = []
      elif (self.train_stage == (len(self.train_data_ratio_list)-2)):
        self.train_stage += 1
        print('make Ensemble job')
        self.__makeEnsembleJob(self.trained_job_list)
      # self.trained_job_list = []
    self.startWorkers()
    if (self.job_queue.qsize() == 0) & (self.train_stage == (len(self.train_data_ratio_list) -1)):
      print('End Ensemble.')
      self.__makeResultDataFrame()

    # print('process Close()')
    # proc.close()

  def __makeTrainedResultJsonData(self, _project_name):
    result_path = definitions.getProjectResultsPath(_project_name)
    file_path_list = glob.glob(str(result_path) + '/*_*.json')
    json_result_list = []
    for file_path in file_path_list:
      if 'meta_info' in file_path:
        continue
      json_data = utils.getJsonFromFile(file_path)
      json_data[KEY_FEATURE_MIS_NAME] = json_data[KEY_FEATURE_MIS_NAME_LIST][json_data[KEY_FEATURE_MIS_NAME]]
      del(json_data[KEY_FEATURE_MIS_NAME_LIST])
      json_data[KEY_FEATURE_OUT_NAME] = json_data[KEY_FEATURE_OUT_NAME_LIST][json_data[KEY_FEATURE_OUT_NAME]]
      del(json_data[KEY_FEATURE_OUT_NAME_LIST])
      json_data[KEY_FEATURE_ADD_NAME] = json_data[KEY_FEATURE_ADD_NAME_LIST][json_data[KEY_FEATURE_ADD_NAME]]
      del(json_data[KEY_FEATURE_ADD_NAME_LIST])
      json_data[KEY_FEATURE_SCA_NAME] = json_data[KEY_FEATURE_SCA_NAME_LIST][json_data[KEY_FEATURE_SCA_NAME]]      
      del(json_data[KEY_FEATURE_SCA_NAME_LIST])
      del(json_data[KEY_FEATURE_SEL_NAME])
      del(json_data[KEY_FEATURE_SEL_NAME_LIST])
      del(json_data[KEY_FEATURE_SEL_RATE_NAME])

      del(json_data[KEY_FEATURE_ADD_NAME])
      del(json_data[KEY_FEATURE_SEL_COL_LIST])
      json_result_list.append(json_data)
    return json_result_list

  def __writeToServerProjectDetail(self, _job):
    print('__writeToServerProjectDetail')
    project_name = _job.project_name    
    column_list = _job.column_list
    column_target = _job.column_target
    eda_path = de_pdprofiling.getVisualizerHtmlFilePath(project_name)
    out_path = df_outlier.getDataframeHtmlFilePath(project_name)  
    train_results = self.__makeTrainedResultJsonData(project_name)
    detail_data = http_request.makeProjectDetailData(project_name, column_list, column_target, eda_path, out_path, train_results)
    id, data = http_request.getProjectDetailIdnData(project_name)
    if id == -1:
      http_request.postHttp(http_request.PREFIX_DETAIL, detail_data)
    else:
      http_request.putHttp(http_request.PREFIX_DETAIL, id, detail_data)

  def __writeProjectInfo(self, _job):
    if (self.job_best):
      if self.job_best.score > _job.score:
        self.job_best = _job
    else:
      self.job_best = _job
    info = project.ProjectMetaInfo(self.job_best.project_name, self.job_best.problem_type, self.job_best.model.metrics_name, self.job_best.score)
    info_dic = info.getDictionary()
    id, data = http_request.getProjectIdnData(self.job_best.project_name)    
    if id == -1:
      http_request.postHttp(http_request.PREFIX_INFO, data)
    else:
      data['best_loss'] = self.job_best.score
      http_request.putHttp(http_request.PREFIX_INFO, id, data)

    filepath = definitions.getProejctInfoFilePath(self.job_best.project_name)
    utils.writeJsonToFile(info_dic, filepath)

  def __makeResultDataFrame(self):
    modelname_list = []
    dataratio_list = []
    f_missing_list = []
    f_outlier_list = []
    f_add_list = []
    f_scaler_List = []
    f_selection_list = []
    f_select_rate = []
    score_list = []
    for job in self.trained_job_list:
      modelname_list.append(job.model.model_name)
      dataratio_list.append(job.data_ratio)
      f_missing_list.append(job.best_params[KEY_FEATURE_MIS_NAME_LIST][job.best_params[KEY_FEATURE_MIS_NAME]])
      f_outlier_list.append(job.best_params[KEY_FEATURE_OUT_NAME_LIST][job.best_params[KEY_FEATURE_OUT_NAME]])
      f_add_list.append(job.best_params[KEY_FEATURE_ADD_NAME_LIST][job.best_params[KEY_FEATURE_ADD_NAME]])
      f_scaler_List.append(job.best_params[KEY_FEATURE_SCA_NAME_LIST][job.best_params[KEY_FEATURE_SCA_NAME]])
      f_selection_list.append(job.best_params[KEY_FEATURE_SEL_NAME_LIST][job.best_params[KEY_FEATURE_SEL_NAME]])
      f_select_rate.append(job.best_params[KEY_FEATURE_SEL_RATE_NAME])
      score_list.append(job.score)
    dic_results = {
      'model': modelname_list,
      'data_ratio': dataratio_list,
      KEY_FEATURE_MIS_NAME: f_missing_list,
      KEY_FEATURE_OUT_NAME: f_outlier_list,
      KEY_FEATURE_ADD_NAME: f_add_list,
      KEY_FEATURE_SCA_NAME: f_scaler_List,
      KEY_FEATURE_SEL_NAME: f_selection_list,
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



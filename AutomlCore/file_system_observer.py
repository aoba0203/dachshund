import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils import definitions
import os
import pandas as pd
from eda import de_pdprofiling, df_outlier
from training.train_manager import TrainManager

class FileSystemWatcher:
  WATCHING_FOLDER = definitions.getWatchingFolder()  
  def __init__(self):
    print('__init__ watching')
    self.observer = Observer()

  def run(self):
    print('run Watching!')
    event_handler = Handler()
    self.observer.schedule(event_handler, self.WATCHING_FOLDER, recursive=True)
    self.observer.start()
    print('Start Watching!')
    try:
      while True:
        time.sleep(5)
    except:
      self.observer.stop()
      print('Error')
    self.observer.join()

class Handler(FileSystemEventHandler):
  @staticmethod
  def on_any_event(event):
    if event.is_directory:
      return None
    elif event.event_type == 'created':
      file_path = event.src_path
      file_name, file_extend = os.path.splitext(os.path.basename(file_path))
      print('created: ', file_path, file_name, file_extend)
      if file_extend == '.csv':
        trainAutoMl(file_path, file_name)
    elif event.event_type == 'modified':
      file_path = event.src_path
      file_name, file_extend = os.path.splitext(os.path.basename(file_path))
      print('modified: ', file_path, file_name, file_extend)

def __getProblemType(_problem_char):
  if _problem_char == 'c':
    return definitions.PROBLEM_TYPE_CLASSIFICATION
  elif _problem_char == 'r':
    return definitions.PROBLEM_TYPE_REGRESSION
  return definitions.PROBLEM_TYPE_CLASSIFICATION

def trainAutoMl(file_path, file_name):
  # path_train_file = 'sample.csv'
  path_train_file = file_path    
  project_name = file_name
  c, target, name = file_name.split('_')
  problem_type = __getProblemType(c)
  profiler = de_pdprofiling.PdProfiling(project_name, path_train_file)
  profiler.makeVisualizerHtmlFile()

  outliers = df_outlier.DfOutlier(project_name, path_train_file)
  outliers.makeDataframeHtmlFile()

  df = pd.read_csv(path_train_file)
  # df = df.sample(int(len(df) * 0.5))
  t = TrainManager(problem_type, project_name, df, target)
  t.startWorkerAdmin()
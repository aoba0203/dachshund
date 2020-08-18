#%%
import os
from utils import definitions, utils
from utils.definitions import KEY_PROJECT_NAME, KEY_PROJECT_PROBLEM_TYPE, KEY_PROJECT_METRICS, KEY_PROJECT_LOSS

class ProjectMetaInfo:
  def __init__(self, _project_name, _problem_type, _metrics_name, _best_loss):
    self.project_name = _project_name
    self.problem_type = _problem_type
    self.metrics_name = _metrics_name
    self.best_loss = _best_loss
  
  def getDictionary(self):
    return {
      KEY_PROJECT_NAME: self.project_name,
      KEY_PROJECT_PROBLEM_TYPE: self.problem_type,
      KEY_PROJECT_METRICS: self.metrics_name, 
      KEY_PROJECT_LOSS: self.best_loss,
    }

def getProjectList():
  info_list = []
  project_name_list = os.listdir(definitions.getResultsPath())
  for project_name in project_name_list:
    if not os.path.isdir(os.path.join(definitions.getResultsPath(), project_name)):
      continue
    filepath = definitions.getProejctInfoFilePath(project_name)
    info_list.append(utils.getJsonFromFile(filepath))
  return info_list


# %%

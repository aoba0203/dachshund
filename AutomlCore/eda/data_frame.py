from abc import ABCMeta, abstractclassmethod

class DataFrame:
  __metaclass__ = ABCMeta
  def __init__(self, _project_name, _csv_filepath):
    self.project_name = _project_name
    self.csv_filepath = _csv_filepath    
    self.__htmlfile_path = None

  @abstractclassmethod
  def getDataframeHtmlFilePath(self):
    pass

  @abstractclassmethod
  def makeDataframeHtmlFile(self):
    pass

  

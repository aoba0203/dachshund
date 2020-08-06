#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
import pandas_profiling
import data_explore
# from . import data_explore
from utils import definitions
#%%
# import sys, importlib
# from pathlib import Path

# def import_parents(level=1):
#     global __package__
#     file = Path(__file__).resolve()
#     parent, top = file.parent, file.parents[level]
    
#     sys.path.append(str(top))
#     try:
#         sys.path.remove(str(parent))
#     except ValueError: # already removed
#         pass

#     __package__ = '.'.join(parent.parts[len(top.parts):])
#     importlib.import_module(__package__) # won't be needed after that
# import_parents()

#%%
class PdProfiling(data_explore.Visualizer):
  def getVisualizerHtmlFilePath(self):
    # eda_path = definitions.getEdaPath()
    # results_path = os.path.join(eda_path, 'results')
    path_result = definitions.getProjectResultsPath(self.project_name)
    if os.path.exists(path_result) == False:
      os.makedirs(path_result)
    return os.path.join(path_result, (self.project_name + '-pandas_profiling.html'))


  def makeVisualizerHtmlFile(self):
    df = pd.read_csv(self.csv_filepath)
    # report = df.profile_report(minimal=True)
    report = df.profile_report()
    report.to_file(output_file= self.getVisualizerHtmlFilePath())

# %%
if __name__ == '__main__':
  rootPath = definitions.getProjectRootPath()
  csv_path = os.path.join(rootPath, 'sample.csv')
  print(csv_path)
  d = PdProfiling('testPjt', csv_path)
  d.makeVisualizerHtmlFile()
  print(d.getVisualizerHtmlFilePath())

# %%

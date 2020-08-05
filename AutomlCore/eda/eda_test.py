import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
from utils import definitions
# import redgreenunittest as unittest
import unittest
import de_pdprofiling
import df_outlier

class EdaTests(unittest.TestCase):
  def setUp(self):
    self.project_name = 'UnitTest'
    self.root_path = definitions.getProjectRootPath()
    self.csv_path = os.path.join(self.root_path, 'sample_small.csv')
  
  def testPdProfiling(self):
    profiling = de_pdprofiling.PdProfiling(self.project_name, self.csv_path)
    self.filepath_profiling = profiling.getVisualizerHtmlFilePath()
    profiling.makeVisualizerHtmlFile()
    self.assertTrue(os.path.exists(self.filepath_profiling))
    pdprofiling_size = os.path.getsize(self.filepath_profiling)
    self.assertGreater(pdprofiling_size, 0)
    os.remove(self.filepath_profiling)
  
  def testOutlier(self):
    outlier = df_outlier.DfOutlier(self.project_name, self.csv_path)
    self.filepath_outlier = outlier.getDataframeHtmlFilePath()    
    outlier.makeDataframeHtmlFile()
    df = pd.read_html(self.filepath_outlier)
    self.assertTrue(os.path.exists(self.filepath_outlier))
    self.assertGreater(len(df), 0)
    os.remove(self.filepath_outlier)
  
  def tearDown(self):
    pass    

if __name__ == '__main__':
  unittest.main()

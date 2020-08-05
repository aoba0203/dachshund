#%%
import pandas as pd
import pandas_profiling
from . import eda

class PdProfiling(eda.EDA):
  def makeEdaHtmlFile(self):
    df = pd.read_csv(self.csv_file)    
    profile = df.profile_report()
    
    profile.to_file(output_file='pandas_profile.html')

  def getHtmlFile(self):
    pass

d = PdProfiling('test111.csv')
d.makeEdaHtmlFile()

# %%
from .. import definitions
print(definitions.getProjectRootPath())

# %%

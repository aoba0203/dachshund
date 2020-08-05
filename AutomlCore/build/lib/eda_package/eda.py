class EDA:
  def __init__(self, csv_filepath):
    self.csv_filepath = csv_filepath
    self.__htmlfile_path = None

  def makeEdaHtmlFile(self):
    pass

  def getHtmlFileName(self, _class_name):
    return 'eda_' + _class_name + '.html'

  def getHtmlFile(self):
    pass

def test(value):
  print(value)

if __name__ == '__main__':
  pass

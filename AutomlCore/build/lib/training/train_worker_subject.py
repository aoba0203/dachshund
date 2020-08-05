
from abc import ABCMeta, abstractclassmethod


class WorkerSubject:
  __metaclass__ = ABCMeta
  EVENT_JOB_END = 100
  EVENT_JOB_FAIL = 101

  @abstractclassmethod
  def registerObserver(self):
    pass

  @abstractclassmethod
  def unregisterObserver(self):
    pass

  @abstractclassmethod
  def notifyObservers(self):
    pass

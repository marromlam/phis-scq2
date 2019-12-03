# -*- coding: utf-8 -*-

import os
import pyopencl as cl

class Badjanak(object):
  """docstring for Badjanak."""

  def __init__(self,  path, interface, context, queue, **kwargs):
    #super(Badjanak, self).__init__()

    # Path to kernel file
    self.path = path
    self.interface = interface
    self.context = context
    self.queue = queue

    self.compile_flags = {
      "DEBUG":           "0",# no prints
      "DEBUG_EVT":       "5",# number of events to debug
      "USE_TIME_ACC":    "0",# NO  time acceptance
      "USE_TIME_OFFSET": "0",# NO  time offset
      "USE_TIME_RES":    "0",# USE time resolution
      "USE_PERFTAG":     "1",# USE perfect tagging
      "USE_TRUETAG":     "0",# NO  true tagging
      "NKNOTS":          "7",
      "NTIMEBINS":       "8",
      "SIGMA_T":         "0.15",
      "KNOTS":           "{0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00}",
      "NMASSBINS":       "6",
      "X_M":             "{990, 1008, 1016, 1020, 1024, 1032, 1050}",
      "TRISTAN":         "{1,1,1,0,0,0,1,0,0,0}"
    }


    # Assign config items
    for configurable in kwargs.keys():
      if configurable in self.compile_flags.keys():
        self.compile_flags[configurable] = kwargs[configurable]
      else:
        print(configurable+' is not in compile_flags!.')

    self.compileCL()
    self.getKernels()

  def updateProperty(self,property,value):
    setattr(self, property, value)
    self.compileCL()
    self.getKernels()

  def compileCL(self):
    kernel_path = os.path.join(self.path,'Badjanak.cl')
    Badjanak = cl.Program(self.context,
                           open(kernel_path,"r").read()
                           .format(**self.compile_flags)
                          ).build(options=["-I "+self.path])
    return Badjanak

  def getKernels(self):
    shit = 'cl'
    if 'cl' == shit:
      try:
        self.__Badjanak = self.compileCL()
        items = self.__Badjanak.kernel_names
        if isinstance(items,str):
          setattr(self, 'k'+items[2:], self.__Badjanak.__getattr__(items))
        else:
          for item in items:
            setattr(self, 'k'+item[2:], self.__Badjanak.__getattr__(item))
      except:
        print('Error!')

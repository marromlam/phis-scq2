import re

def cammel_case_split(str):
  return re.sub('([A-Z][a-z]+)',r' \1',re.sub('([A-Z]+)',r' \1', str)).split()

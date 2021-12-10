import re
import hjson


def printsec(string):
  print(f"\n{80*'='}\n{string}\n{80*'='}\n")

def printsubsec(string):
  print(f"\n{string}\n{80*'='}\n")

def cammel_case_split(str):
  return re.sub('([A-Z][a-z]+)',r' \1',re.sub('([A-Z]+)',r' \1', str)).split()


def cuts_and(*args):
  result = list(args)
  result = [a for a in args if a]
  return '(' + ') & ('.join(result) + ')'

def cuts_or(*args):
  result = list(args)
  result = [a for a in args if a]
  return '(' + ') | ('.join(result) + ')'

# def cammel_parser(input, verbose=False):
#   if verbose:
#     print(f"{'input':>25} : {input}")
#   #input = input.split('_')
#   # if len(input) > 1:
#   #   print("Error")
#   #input = input[0]
#   #pinput = cammel_case_split(input)
#   CUT = None
#   magnet = None; binned = None; temp = None
#   # Place cuts on magnet polarity
#   if "MagUp" in input:
#     magnet = 'Up'
#     input = ''.join(input.split('MagUp'))
#   elif "MagDown" in input:
#     magnet = 'Down'
#     input = ''.join(input.split('MagDown'))
#   if magnet:
#     CUT = cuts_and(magnet,CUT)
#   # Place cuts onf binned variables for fit check
#   if "Bin" in input:
#     temp = input.split('Bin'); #print(temp)
#     if 'pt' in temp[1]:
#       binned = bin_vars['pt'][int(temp[1][len('pt')])-1]
#       input = ''.join(input.split(f"Binpt{temp[1][len('pt')]}"))
#     elif 'eta' in temp[1]:
#       binned = bin_vars['eta'][int(temp[1][len('eta')])-1]
#       input = ''.join(input.split(f"Bineta{temp[1][len('eta')]}"))
#     elif 'sigmat' in temp[1]:
#       binned = bin_vars['sigmat'][int(temp[1][len('sigmat')])-1]
#       input = ''.join(input.split(f"Binsigmat{temp[1][len('sigmat')]}"))
#     else:
#       print("Error")
#   if binned:
#     CUT = cuts_and(binned,CUT)
#   if verbose:
#     print(f"{'main':>25} : {input}")
#     print(f"{'magnet cut':>25} : {magnet}")
#     print(f"{'binned cut':>25} : {binned}")
#     print(f"{'CUT':>25} : {CUT}")
#   return CUT

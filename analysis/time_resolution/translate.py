# translate
#
#

__all__ = []
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


import argparse
import json
import os
from array import array
import numpy as np
import scipy.linalg as la
from math import log, exp, sqrt
import ipanema
import uncertainties as unc
# from compute_dilution import sigma_eff_error

DM = 17.74


def corrected_per_bin(prompt_data, prompt_mc, signal_mc, deltaMs):
    new_values = []
    new_dilutions = []
    for i in range(len(prompt_data)):
        correction = signal_mc[i]/prompt_mc[i]
        new_dilution = correction*prompt_data[i]
        new_sigma_eff = sqrt(-2*log(new_dilution))*1./deltaMs
        new_dilutions.append(new_dilution)
        new_values.append(new_sigma_eff)
    return [new_values, new_dilutions]


def corrected_error_per_bin(new_values, prompt_data, prompt_mc, signal_mc, deltaMs):
    new_errors = []
    for i in range(len(prompt_data)):
        error = sqrt(pow(prompt_data[i]["Error"]/prompt_data[i]["Value"], 2) + pow(signal_mc[i]["Error"] /
                     signal_mc[i]["Value"], 2) + pow(prompt_mc[i]["Error"]/prompt_mc[i]["Value"], 2))*new_values[i]
        sigma_eff_err = sigma_eff_error(
            delta_m=deltaMs, D=new_values[i], Derr=error)
        new_errors.append(sigma_eff_err)
    return new_errors


def corrected_error_dilution(new_values, prompt_data, prompt_mc, signal_mc, deltaMs):
    new_errors = []
    for i in range(len(prompt_data)):
        error = sqrt(pow(prompt_data[i]["Error"]/prompt_data[i]["Value"], 2) + pow(signal_mc[i]["Error"] /
                     signal_mc[i]["Value"], 2) + pow(prompt_mc[i]["Error"]/prompt_mc[i]["Value"], 2))*new_values[i]
        print(prompt_data[i]["Error"], prompt_mc[i]
              ["Error"], signal_mc[i]["Error"])
        new_errors.append(error)
    return new_errors


def effective_dilution(new_values, new_values_errors, num_prompt_mc, num_signal_mc, num_prompt_data):
    Sum = 0
    sum_num = 0
    new_numbers = []
    for i in range(len(new_values)):
        print(i)
        correction = num_signal_mc[i]/num_prompt_mc[i]
        new_num = correction*num_prompt_data[i]
        new_numbers.append(new_num)
        Sum = Sum + new_num*new_values[i]*new_values[i]
        print(new_values_errors[i])
    D_eff = sqrt(Sum/sum(new_numbers))
    D_eff_err = sqrt(pow(1./(2.*D_eff*sum(new_numbers)), 2)*(pow(D_eff, 2)/sum(new_numbers) + sum([pow(i, 2)/j for i, j in zip(
        new_values, new_numbers)]) + 2*sum([pow(i*j*k, 2) for i, j, k in zip(new_values, new_numbers, new_values_errors)])))
    return [D_eff, D_eff_err]


def calibrate_line(line_signalmc, line_promptmc, line_promptdata, bins):
    calibrated_line = TF1(
        "line", "([0]+[1]*x)*([2]+[3]*x)/([4]+[5]*x)", float(bins[0]), float(bins[-1]))
    calibrated_line.SetParameter(0, line_signalmc[0]["Value"])
    calibrated_line.SetParameter(1, line_signalmc[1]["Value"])
    calibrated_line.SetParameter(2, line_promptdata[0]["Value"])
    calibrated_line.SetParameter(3, line_promptdata[1]["Value"])
    calibrated_line.SetParameter(4, line_promptmc[0]["Value"])
    calibrated_line.SetParameter(5, line_promptmc[1]["Value"])

    return calibrated_line


def translate(prompt_mc, mc, prompt_data, sigma_edges):
    print("Prompt MC params")
    print(prompt_mc)
    print("Signal MC params")
    print(mc)
    print("Prompt RD params")
    print(prompt_data)

    dil_mc  = ipanema.Parameters.build(mc, mc.find('dcorr.*'))
    dil_mcp = ipanema.Parameters.build(prompt_mc, prompt_mc.find('dcorr.*'))
    dil_rdp = ipanema.Parameters.build(prompt_data, prompt_data.find('dcorr.*'))

    print("Signal MC")
    print(dil_mc)
    print("Signal MC Prompt")
    print(dil_mcp)
    print("Signal RD Prompt")
    print(dil_rdp)
    dil_mc  = dil_mc.uvaluesdict()
    dil_mcp = dil_mcp.uvaluesdict()
    dil_rdp = dil_rdp.uvaluesdict()

    num_mc  = ipanema.Parameters.build(mc, mc.find('neff.*'))
    num_mcp = ipanema.Parameters.build(prompt_mc, prompt_mc.find('neff.*'))
    num_rdp = ipanema.Parameters.build(prompt_data, prompt_data.find('neff.*'))
    num_mc  = num_mc.uvaluesdict()
    num_mcp = num_mcp.uvaluesdict()
    num_rdp = num_rdp.uvaluesdict()

    dil_rd = {}
    sig_rd = {}
    for k, tmc, tmcp, trdp in zip(dil_mc.keys(), dil_mc.values(),
                                  dil_mcp.values(), dil_rdp.values()):
        _dil = k
        _seff = k.replace('dcorr', 'seff')
        dil_rd[_dil] = trdp * (tmc / tmcp)
        sig_rd[_seff] = unc.wrap(sqrt)(-2*unc.wrap(log)(dil_rd[k]))/DM
    print(dil_rd)
    print(sig_rd)

    pars = ipanema.Parameters.clone(prompt_data)
    for k, v in dil_rd.items():
        pars[k].set(value=v.n, stdev=v.s)
    for k, v in sig_rd.items():
        pars[k].set(value=v.n, stdev=v.s)

    return pars


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Determine 1st order polynomial fit parameters.')
    p.add_argument('--mode')
    p.add_argument('--timeres')
    p.add_argument('--in-json-mc')
    p.add_argument('--in-json-mc-prompt')
    p.add_argument('--in-json-rd-prompt')
    p.add_argument('--out-json-rd')
    p.add_argument('--out-plots', help='Directory to save plots')
    args = vars(p.parse_args())

    timeres_binning = [0.010, 0.021, 0.026, 0.032,
                       0.038, 0.044, 0.049, 0.054, 0.059, 0.064, 0.08]
    time_range = [-4, 10]

    mc = ipanema.Parameters.load(args['in_json_mc'])
    mcprompt = ipanema.Parameters.load(args['in_json_mc_prompt'])
    rdprompt = ipanema.Parameters.load(args['in_json_rd_prompt'])

    pars = translate(mcprompt, mc, rdprompt, timeres_binning)
    pars.dump(args['out_json_rd'])


# vim: fdm=marker ts=2 sw=2 sts=2 sr et

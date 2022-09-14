from __future__ import print_function, division
import os
import argparse
from glob import glob
from array import array
from math import pi, sqrt, sin, cos, atan2, exp
# from ROOT import gStyle
# from ROOT import kBlack, kBlue, kRed
# from ROOT import TGraphErrors as Graph
# from ROOT import TCanvas, TLatex, TLine, kDotted
# from ROOT import RooProduct, RooArgSet, RooArgList
# from ROOT import TFile, RooDataSet, TObject, kFullDotLarge
# from P2VV.RooFitWrappers import __dref__
# from P2VV.RooFitWrappers import RooObject, RealVar, Category
# from P2VV.Load import RooFitOutput, LHCbStyle


NBINS = 8
TIMEFRACS = [(float(it)/100.) for it in range(100)] # ['0.00', '0.50'] # ['0.00', '0.25', '0.50', '0.75']
ASYMMAX = 0.07 # plotting range
DELTAM = 17.70568
OSCPERIOD = 2. * pi / DELTAM
BINWIDTH = 2. * pi / DELTAM / float(NBINS)
BINOFFSET = 0.3 / BINWIDTH
PERIODSHIFT = -1.


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-in', help='Path to the dataset')
    parser.add_argument('--create-asym-dataset', default=False)
    parser.add_argument('--dataset-out', default='output/data/asymmetry_data.root', help='Path to the output asymmetry dataset')
    parser.add_argument('--plot-pdf-int', default=False, action='store_true', help='plot integrated-PDF asymmetry in time bins')
    parser.add_argument('--plot-pdf', default=False, action='store_true', help='plot PDF asymmetry in time bins')
    parser.add_argument('--apply-dil-weights', default=False, action='store_true', help='correct for dilution')
    parser.add_argument('--apply-ang-weights', default='', help='probably no, option para_perp')
    parser.add_argument('--dilution-cut', default=0.01, help='what do i know..')
    parser.add_argument('--path-to-pdf', default='output/fit/pdf_vals/pdfVals_08bins_6p763_b*_p*_f100p.par')
    parser.add_argument('--path-to-plots', default='output/fit/figs/')
    return parser


def read_pdf_values(filePathsString):
    '''Function to read files with PDF values'''
    filePaths = glob(filePathsString)
    # _pdfIntVals = dict(plus=[0.] * NBINS, minus=[0.] * NBINS)
    _pdfVals = dict(plus=[dict(('{0:.2f}'.format(frac), 0.) for frac in TIMEFRACS) for bin in range(NBINS)],
                    minus=[dict(('{0:.2f}'.format(frac), 0.) for frac in TIMEFRACS) for bin in range(NBINS)])
    fileCount = 0
    for path in filePaths:
        valFile = open(path)
        bin = -1
        while True:
            line = valFile.readline().split()
            if not line: break
            if bin < 0:
                assert line[0] == 'bin'
                bin = int(line[1])
                line = valFile.readline().split()
            # elif line[0] == 'integrals':
            #     plusIntVal = float(line[2])
            #     minIntVal  = float(line[3])
            else:
                _pdfVals['plus'][bin][line[0]] += float(line[3])
                _pdfVals['minus'][bin][line[0]] += float(line[4])

        # _pdfIntVals['plus'][bin] += plusIntVal
        # _pdfIntVals['minus'][bin] += minIntVal

        valFile.close()
        fileCount += 1
    print('plotAsymmetry: read data from %d PDF values files (%s)' % (fileCount, filePathsString))
    return  (_pdfVals)


def set_graph_atts(graph, colour, markerSize, minimum, maximum):
    '''Function to set graph attributes'''
    graph.SetMarkerStyle(kFullDotLarge)
    graph.SetLineColor(colour)
    graph.SetMarkerColor(colour)
    graph.SetLineWidth(3)
    graph.SetMarkerSize(markerSize)
    graph.GetXaxis().SetTitle('Decay time (modulo 2#pi/#Deltam_{s}) [ps]')
    graph.GetYaxis().SetTitle('B/#bar{B}-tag asymmetry')
    graph.GetXaxis().SetTitleOffset(1.2)
    graph.GetYaxis().SetTitleOffset(1.2)
    graph.GetXaxis().SetLimits((PERIODSHIFT + BINOFFSET / float(NBINS)) * OSCPERIOD,
                         (PERIODSHIFT + (1. + BINOFFSET / float(NBINS))) * OSCPERIOD)
    graph.SetMinimum(minimum)
    graph.SetMaximum(maximum)
    graph.GetXaxis().SetNdivisions(10, 5, 0, True)
    graph.GetYaxis().SetNdivisions(5, 5, 0, True)


def plot_cp_asymmetry(dataset_in, create_asym_dataset, dataset_out, plot_pdf_int,
                      plot_pdf, apply_dil_weights, apply_ang_weights, dilution_cut,
                      path_to_pdf, path_to_plots):
    ws = RooObject(workspace = 'JpsiphiWorkspace').ws()

    if create_asym_dataset:
        # read data set from file
        dataFile = TFile.Open(dataset_in)
        dataSet = dataFile.Get('JpsiKK')
        dataFile.Close()
        dataSet.Print()

        # get set of observables in data set
        obsSet = RooArgSet(dataSet.get())
        sw = RealVar(Name='sw', Value=1.)
        dilution = RealVar(Name='dilution', Value=1.)
        asymCat = Category(Name='asymCat', States=dict(plus=+1, minus=-1))
        obsSet.add(__dref__(sw))
        obsSet.add(__dref__(dilution))
        obsSet.add(__dref__(asymCat))
        if apply_ang_weights:
            angWeight = RealVar(Name='angWeight_%s' % apply_ang_weights, Value=1.)
            obsSet.add(__dref__(angWeight))

        # create data set with events in two asymmetry categories
        print('plotAsymmetry: creating dataset with events in two asymmetry categories')
        dataSetAsym = RooDataSet('asymData', 'asymData', obsSet)
        for evSet in dataSet:
            # get dilution from resolution
            sigmat = evSet.getRealValue('sigmat')
            resDil = exp(-0.5 * DELTAM**2 * sigmat**2)

            # get dilution from tagging
            tagCat = (int(abs(evSet.getCatIndex('tagos_dec_new')) > 0), int(abs(evSet.getCatIndex('B_SSKaonLatest_TAGDEC')) > 0))
            tags = (evSet.getCatIndex('tagos_dec_new'), evSet.getCatIndex('B_SSKaonLatest_TAGDEC'))

            tag = 0
            tagDil = 0.
            if tagCat[0] == 1:
                tag = tags[0]
                tagDil = 1. - 2. * evSet.getRealValue('tagos_eta_new')
                if tagCat[1] == 1:
                    tagDilAlt = 1. - 2. * evSet.getRealValue('B_SSKaonLatest_TAGETA')
                    if tags[0] == tags[1]:
                        tagDil = (tagDil + tagDilAlt) / (1 + tagDil * tagDilAlt)
                    else :
                        tagDil = (tagDil - tagDilAlt) / (1 - tagDil * tagDilAlt)
                        if tagDil < 0.:
                            tag = -tag
                            tagDil = -tagDil
            elif tagCat[1] == 1:
                tag = tags[1]
                tagDil = 1. - 2. * evSet.getRealValue('B_SSKaonLatest_TAGETA')
            else: continue


            if resDil * tagDil < dilution_cut: continue

            # set signal weight
            sw.setVal(dataSet.weight())

            # set tagging observables
            dilution.setVal((resDil * tagDil))
            asymCat.setIndex(tag)

            # calculate angular weight
            if apply_ang_weights:
                ctk = evSet.getRealValue('helcosthetaK')
                ctl = evSet.getRealValue('helcosthetaL')
                phi = evSet.getRealValue('helphi')
            if apply_ang_weights == 'ang':
                angWeight.setVal(2. - 5. * (1. - ctl**2) * sin(phi)**2)
            elif apply_ang_weights == 'para_perp':
                angWeight.setVal((9. - 15. * ctk**2) * (1. - ctl**2) * sin(2. * phi))
                #angWeight.setVal( ( 1. - ctl**2 ) * sin(2. * phi) )

            # add event to dataset
            dataSetAsym.add(obsSet)
        dataSetAsym_to_save = dataSetAsym.Clone()
        dataFile = TFile.Open(dataset_out, 'RECREATE')
        dataFile.Add(dataSetAsym_to_save)
        dataFile.Write(dataset_out, TObject.kOverwrite)
        dataFile.Close()
    else :
        # read data set with events in two asymmetry categories
        print('plotAsymmetry: reading dataset with events in two asymmetry categories')
        dataFile = TFile.Open(dataset_out)
        dataSetAsym = dataFile.Get('asymData')
        dataFile.Close()
    dataSetAsym.Print()

    # create weighted data set
    obsSet = RooArgSet(dataSetAsym.get())
    prodList = RooArgList(obsSet.find('sw'))
    if apply_dil_weights: prodList.add(obsSet.find('dilution'))
    if apply_ang_weights: prodList.add(obsSet.find('angWeight_%s' % apply_ang_weights))
    weightVar = RooProduct('weightVar', 'weightVar', prodList)
    weightVar = dataSetAsym.addColumn(weightVar)
    obsSet.add(weightVar)

    dataSetAsymW = RooDataSet('asymDataW', 'asymDataW', obsSet, Import=dataSetAsym, WeightVar=('weightVar', True))
    del dataSetAsym
    ws.put(dataSetAsymW)
    del dataSetAsymW
    dataSetAsymW = ws['asymDataW']
    obsSet = RooArgSet(dataSetAsymW.get())
    dataSetAsymW.Print()

    # get sums of weights
    sumW = dict(plus=0., minus=0.)
    for evSet in dataSetAsymW:
        if evSet.getCatIndex('asymCat') == 1:
            sumW['plus'] += dataSetAsymW.weight()
        else:
            sumW['minus'] += dataSetAsymW.weight()
    assert abs(dataSetAsymW.sumEntries() - sumW['plus'] - sumW['minus']) < 1.e-5
    if apply_ang_weights:
        ASumW = 2. * sumW['plus'] - (sumW['plus'] + sumW['minus'])
    else:
        ASumW = 2. * sumW['plus'] / (sumW['plus'] + sumW['minus']) - 1.

    # create arrays of time bins
    timeArr = array('d', [(PERIODSHIFT + (float(it) + BINOFFSET + 0.5) / float(NBINS)) * OSCPERIOD\
                          for it in range(NBINS + 1)])
    print(timeArr)
    timeErrArr = array('d', [0.5 / float(NBINS) * OSCPERIOD] * (NBINS + 1))
    timeArrPdf = array('d', [(PERIODSHIFT + (float(it) + BINOFFSET + float(frac)) / float(NBINS)) * OSCPERIOD\
                             for it in range(NBINS + 1) for frac in TIMEFRACS]\
                             + [(PERIODSHIFT + 1. + (1. + BINOFFSET + float(TIMEFRACS[0])) / float(NBINS)) * OSCPERIOD])
    timeErrArrPdf = array('d', [0.] * len(timeArrPdf))

    if plot_pdf_int or plot_pdf:
        (pdfVals) = read_pdf_values(path_to_pdf)

    Graph.setAtts = set_graph_atts

    gStyle.SetColorModelPS(1)
    pdfGraph = None
    if plot_pdf:
        # plot PDF asymmetry in time bins
        vals = pdfVals
        valsTot = pdfVals

        pdfArr = array( 'd', [(pVals["{0:.2f}".format(frac)] - mVals["{0:.2f}".format(frac)]) / (pValsTot["{0:.2f}".format(frac)] + mValsTot["{0:.2f}".format(frac)])\
                              for pVals, mVals, pValsTot, mValsTot in zip(vals['plus'], vals['minus'], valsTot['plus'], valsTot['minus'])\
                              for frac in TIMEFRACS])
        for it in range(len(TIMEFRACS) + 1): pdfArr.append(pdfArr[it])
        pdfErrArr = array('d', [0.] * len(pdfArr))
        pdfGraph = Graph(len(timeArrPdf), timeArrPdf, pdfArr, timeErrArrPdf, pdfErrArr)
        pdfGraph.SetName('pdf')

    pdfIntGraph = None
    if plot_pdf_int:
        # plot integrated-PDF asymmetry in time bins
        intVals = pdfIntVals
        pdfIntArr = array('d', [2. * pVal / (pVal + mVal) - 1. for pVal, mVal in zip(intVals['plus'], intVals['minus'])])
        pdfIntArr.append(pdfIntArr[0])
        pdfIntErrArr = array('d', [0.] * len(pdfIntArr))
        pdfIntGraph = Graph(len(timeArr), timeArr, pdfIntArr, timeErrArr, pdfIntErrArr)
        pdfIntGraph.SetName('pdfInt')

    dataGraph = None
    # get data asymmetries in time bins
    eventSums = dict([(var, [0.] * NBINS) for var in ('m0', 'm1', 'n0', 'n1')])
    timeBins = [(1. + float(it) + BINOFFSET) / float(NBINS) * OSCPERIOD for it in range(NBINS)]
    for evSet in dataSetAsymW:
        time = evSet.getRealValue('time')
        bin = 0
        iter = 0
        while time >= float(iter) * OSCPERIOD + timeBins[bin]:
            if bin < len(timeBins) - 1:
                bin += 1
            else :
                bin = 0
                iter += 1

        weight = dataSetAsymW.weight()
        eventSums['m0'][bin] += weight
        eventSums['n0'][bin] += weight**2
        if evSet.getCatIndex('asymCat') == 1:
            eventSums['m1'][bin] += weight
            eventSums['n1'][bin] += weight**2
    assert abs(sumW['plus'] - sum( eventSums['m1'])) < 1.e-5 and abs(sumW['plus'] + sumW['minus'] - sum( eventSums['m0'])) < 1.e-5

    print(sumW['plus'])
    print(sumW['minus'])

    # plot data asymmetry as a function of time
    if apply_ang_weights:
        dataArr = array('d', [2. * m1 - m0 for m0, m1 in zip( eventSums['m0'], eventSums['m1'])])
        dataErrArr = array('d', [sqrt(n0) for n0 in eventSums['n0']])
    else:
        dataArr = array('d', [2. * m1 / m0 - 1. for m0, m1 in zip( eventSums['m0'], eventSums['m1'])])
        dataErrArr = array('d', [2. / m0 * sqrt( n0 * (1. + nu)**2 / 4. - n1 * nu)\
                                  for nu, m0, n0, n1 in zip(dataArr, eventSums['m0'], eventSums['n0'], eventSums['n1'])])
    dataArr.append(dataArr[0])
    dataErrArr.append(dataErrArr[0])
    dataGraph = Graph(len(timeArr), timeArr, dataArr, timeErrArr, dataErrArr)
    dataGraph.SetName('data')

    # set graph attributes
    dummyGraph = Graph(NBINS + 1)
    dummyGraph.SetName('dummy')
    dummyGraph.setAtts(kBlack, 0., -ASYMMAX, +ASYMMAX)
    if dataGraph: dataGraph.setAtts(kBlack, 1., -ASYMMAX, +ASYMMAX)
    if pdfIntGraph: pdfIntGraph.setAtts(kBlue, 1., -ASYMMAX, +ASYMMAX)
    if pdfGraph: pdfGraph.setAtts(kRed if pdfIntGraph else kBlue, 0.1, -ASYMMAX, +ASYMMAX)

    # create line
    dotLine = TLine()
    dotLine.SetLineStyle(kDotted)

    # create label
    label = TLatex()
    label.SetTextAlign(12)
    label.SetTextSize(0.072)

    # draw graphs
    tMin = dummyGraph.GetXaxis().GetXmin()
    tMax = dummyGraph.GetXaxis().GetXmax()
    canv = TCanvas('canv')
    canv.SetLeftMargin(0.18)
    canv.SetRightMargin(0.05)
    canv.SetBottomMargin(0.18)
    canv.SetTopMargin(0.05)
    dummyGraph.Draw('AP')
    horLine = dotLine.DrawLine(tMin, ASumW, tMax, ASumW)
    if dataGraph: dataGraph.Draw('P SAMES')
    if pdfIntGraph: pdfIntGraph.Draw('P SAMES')
    if pdfGraph: pdfGraph.Draw('L SAMES')
    if dataGraph: dataGraph.Draw('P SAMES')
    label.DrawLatex(tMin + 0.77 * (tMax - tMin), 0.76 * ASYMMAX, 'LHCb')
    canv.Print(path_to_plots+'asymPlot_realBins_points.pdf')

    # save graphs and canvas
    plotsROOTFile = TFile.Open(path_to_plots+'asymPlot_realBins_points.root', 'RECREATE')
    plotsROOTFile.Add(dummyGraph)
    plotsROOTFile.Add(horLine)
    if dataGraph: plotsROOTFile.Add(dataGraph)
    if pdfIntGraph: plotsROOTFile.Add(pdfIntGraph)
    if pdfGraph: plotsROOTFile.Add(pdfGraph)
    plotsROOTFile.Add(canv)
    plotsROOTFile.Write(path_to_plots+'asymPlot_realBins_points.root', TObject.kOverwrite)
    plotsROOTFile.Close()


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    plot_cp_asymmetry(**vars(args))


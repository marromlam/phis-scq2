#!/usr/bin/env python
# @(#)root/tmva $Id$
# ------------------------------------------------------------------------------ #
# Project      : TMVA - a Root-integrated toolkit for multivariate data analysis #
# Package      : TMVA                                                            #
# Python script: TMVAClassification.py                                           #
#                                                                                #
# This python script provides examples for the training and testing of all the   #
# TMVA classifiers through PyROOT.                                               #
#                                                                                #
# The Application works similarly, please see:                                   #
#    TMVA/macros/TMVAClassificationApplication.C                                 #
# For regression, see:                                                           #
#    TMVA/macros/TMVARegression.C                                                #
#    TMVA/macros/TMVARegressionpplication.C                                      #
# and translate to python as done here.                                          #
#                                                                                #
# As input data is used a toy-MC sample consisting of four Gaussian-distributed  #
# and linearly correlated input variables.                                       #
#                                                                                #
# The methods to be used can be switched on and off via the prompt command, for  #
# example:                                                                       #
#                                                                                #
#    python TMVAClassification.py --methods Fisher,Likelihood                    #
#                                                                                #
# The output file "TMVA.root" can be analysed with the use of dedicated          #
# macros (simply say: root -l <../macros/macro.C>), which can be conveniently    #
# invoked through a GUI that will appear at the end of the run of this macro.    #
#                                                                                #
# for help type "python TMVAClassification.py --help"                            #
# ------------------------------------------------------------------------------ #

# --------------------------------------------
# Standard python import
import sys
import argparse
import yaml
from array import array
from ROOT import (
    TCut,
    ROOT,
    RDataFrame,
    gROOT,
    TMVA,
    TFile
)


def read_from_yaml(mode, selection_files):
    bdt_dict = dict()
    for file in selection_files:
        with open(file, 'r') as stream:
            bdt_dict.update(yaml.safe_load(stream)[mode])
    return bdt_dict


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal-file', help='Path to the signal file')
    parser.add_argument('--signal-tree-name', default='DecayTree', help='Name of the tree')
    parser.add_argument('--signal-weight', default='1', help='Weight variable')
    parser.add_argument('--background-file', help='Path to the background file')
    parser.add_argument('--background-tree-name', default='DecayTree', help='Name of the tree')
    parser.add_argument('--background-weight', default='1', help='Weight variable')
    parser.add_argument('--output-dir', help='Output ROOT file')
    parser.add_argument('--bdt-vars', nargs='+', required=True, help='Yaml files with selection')
    parser.add_argument('--mode', help='Name of the selection in yaml')
    parser.add_argument('--bdt-method-name', default='BDTG3', help='Choose which BDT to apply')
    parser.add_argument('--year', required=True, help='Year of data taking')
    return parser


def train_tmva(signal_file, signal_tree_name, signal_weight, background_file,
        background_tree_name, background_weight, output_dir, bdt_vars, mode,
        bdt_method_name, year):
    # check ROOT version, give alarm if 5.18
    if gROOT.GetVersionCode() >= 332288 and gROOT.GetVersionCode() < 332544:
        print("*** You are running ROOT version 5.18, which has problems in PyROOT such that TMVA")
        print("*** does not run properly (function calls with enums in the argument are ignored).")
        print("*** Solution: either use CINT or a C++ compiled version (see TMVA/macros or TMVA/examples),")
        print("*** or use another ROOT version (e.g., ROOT 5.19).")
        sys.exit(1)

    # read the cuts and branches from all input files
    bdt_conversion = read_from_yaml(mode, bdt_vars)

    # Output file
    output_file = output_dir+'TMVA.root'
    outputFile = TFile(output_file, 'RECREATE')

    # Create instance of TMVA factory (see TMVA/macros/TMVAClassification.C for more factory options)
    # All TMVA output can be suppressed by removing the "!" (not) in
    # front of the "Silent" argument in the option string
    factory = TMVA.Factory( "TMVAClassification", outputFile,
                            "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" )

    # Set verbosity
    factory.SetVerbose(True)

    (TMVA.gConfig().GetIONames()).fWeightFileDir = output_dir
    dataloader = TMVA.DataLoader("")

    # If you wish to modify default settings
    # (please check "src/Config.h" to see all available global options)
    #    gConfig().GetVariablePlotting()).fTimesRMS = 8.0
    #    gConfig().GetIONames()).fWeightFileDir = "myWeightDirectory"

    # Define the input variables that shall be used for the classifier training
    # note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
    # [all types of expressions that can also be parsed by TTree::Draw( "expression" )]

    mva_vars = dict()
    # assign array to each mva variable
    for var in bdt_conversion.keys():
        mva_vars[var] = array('f', [-999])
        dataloader.AddVariable(var)

    # You can add so-called "Spectator variables", which are not used in the MVA training,
    # but will appear in the final "TestTree" produced by TMVA. This TestTree will contain the
    # input variables, the response values of all trained MVAs, and the spectator variables
    # factory.AddSpectator( "spec1:=var1*2",  "Spectator 1", "units", 'F' )
    # factory.AddSpectator( "spec2:=var1*3",  "Spectator 2", "units", 'F' )

    # Read input data
    sigFile = TFile.Open(signal_file)
    bkgFile = TFile.Open(background_file)

    # Get the signal and background trees for training
    signal = sigFile.Get(signal_tree_name)
    background = bkgFile.Get(background_tree_name)

    # Decide training/testing proportions
    kTrainSubsample = 0.7
    kSigNumTrainEvents = (int)(kTrainSubsample * signal.GetEntries())
    kBkgNumTrainEvents = (int)(kTrainSubsample * background.GetEntries())
    # kNumTrainEvents = 100000;
    kSigNumTestEvents = (int)(signal.GetEntries() - kSigNumTrainEvents)
    kBkgNumTestEvents = (int)(background.GetEntries() - kBkgNumTrainEvents)
    # kNumTestEvents  = 600000;

    print("--- TMVAClassification       : Using signal file: ", signal_file)
    print("--- TMVAClassification       : Using background file: ", background_file)
    print("--- TMVAClassification       : Using number of events: ")
    print("--- TMVAClassification       : Train signal:     ", kSigNumTrainEvents)
    print("--- TMVAClassification       : Train background: ", kBkgNumTrainEvents)
    print("--- TMVAClassification       : Test signal:      ", kSigNumTestEvents)
    print("--- TMVAClassification       : Test background:  ", kBkgNumTestEvents)


    # Global event weights (see below for setting event-wise weights)
    signalWeight = 1.0
    backgroundWeight = 1.0

    # ====== register trees ====================================================
    #
    # the following method is the prefered one:
    # you can add an arbitrary number of signal or background trees
    dataloader.AddSignalTree    ( signal,     signalWeight     )
    dataloader.AddBackgroundTree( background, backgroundWeight )

    # To give different trees for training and testing, do as follows:
    #    factory.AddSignalTree( signalTrainingTree, signalTrainWeight, "Training" )
    #    factory.AddSignalTree( signalTestTree,     signalTestWeight,  "Test" )

    # Use the following code instead of the above two or four lines to add signal and background
    # training and test events "by hand"
    # NOTE that in this case one should not give expressions (such as "var1+var2") in the input
    #      variable definition, but simply compute the expression before adding the event
    #
    #    # --- begin ----------------------------------------------------------
    #
    # ... *** please lookup code in TMVA/macros/TMVAClassification.C ***
    #
    #    # --- end ------------------------------------------------------------
    #
    # ====== end of register trees ==============================================

    # Set individual event weights (the variables must exist in the original TTree)
    #    for signal    : factory.SetSignalWeightExpression    ("weight1*weight2");
    #    for background: factory.SetBackgroundWeightExpression("weight1*weight2");
    if signal_weight:
        dataloader.SetSignalWeightExpression(str(signal_weight))
    if background_weight:
        dataloader.SetBackgroundWeightExpression(str(background_weight))

    # Apply additional cuts on the signal and background sample.
    # example for cut: mycut = TCut( "abs(var1)<0.5 && abs(var2-0.5)<1" )
    #mycuts = TCut(""); # for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
    #mycutb = TCut(""); # for example: TCut mycutb = "abs(var1)<0.5";

    # Here, the relevant variables are copied over in new, slim trees that are
    # used for TMVA training and testing
    # "SplitMode=Random" means that the input events are randomly shuffled before
    # splitting them into training and test samples
    #dataloader.PrepareTrainingAndTestTree( mycutSig, mycutBkg,
    #                                    "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V" )
    mycuts = TCut("")
    dataloader.PrepareTrainingAndTestTree(mycuts, kSigNumTrainEvents, kBkgNumTrainEvents, kSigNumTestEvents, kBkgNumTestEvents, "SplitMode=Random:NormMode=NumEvents:!V")

    # --------------------------------------------------------------------------------------------------

    # ---- Book MVA methods
    #
    # please lookup the various method configuration options in the corresponding cxx files, eg:
    # src/MethoCuts.cxx, etc, or here: http://tmva.sourceforge.net/optionRef.html
    # it is possible to preset ranges in the option string in which the cut optimisation should be done:
    # "...:CutRangeMin[2]=-1:CutRangeMax[2]=1"...", where [2] is the third input variable

    # Cut optimisation
    if bdt_method_name == "Cuts":
        factory.BookMethod( dataloader, TMVA.Types.kCuts, "Cuts",
                            "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart" )

    if bdt_method_name == "CutsD":
        factory.BookMethod( dataloader, TMVA.Types.kCuts, "CutsD",
                            "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart:VarTransform=Decorrelate" )

    if bdt_method_name == "CutsPCA":
        factory.BookMethod( dataloader, TMVA.Types.kCuts, "CutsPCA",
                            "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart:VarTransform=PCA" )

    if bdt_method_name == "CutsGA":
        factory.BookMethod( dataloader, TMVA.Types.kCuts, "CutsGA",
                            "H:!V:FitMethod=GA:CutRangeMin[0]=-10:CutRangeMax[0]=10:VarProp[1]=FMax:EffSel:Steps=30:Cycles=3:PopSize=400:SC_steps=10:SC_rate=5:SC_factor=0.95" )

    if bdt_method_name == "CutsSA":
        factory.BookMethod( dataloader, TMVA.Types.kCuts, "CutsSA",
                            "!H:!V:FitMethod=SA:EffSel:MaxCalls=150000:KernelTemp=IncAdaptive:InitialTemp=1e+6:MinTemp=1e-6:Eps=1e-10:UseDefaultScale" )

    # Likelihood ("naive Bayes estimator")
    if bdt_method_name == "Likelihood":
        factory.BookMethod( dataloader, TMVA.Types.kLikelihood, "Likelihood",
                            "H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" )

    # Decorrelated likelihood
    if bdt_method_name == "LikelihoodD":
        factory.BookMethod( dataloader, TMVA.Types.kLikelihood, "LikelihoodD",
                            "!H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmooth=5:NAvEvtPerBin=50:VarTransform=Decorrelate" )

    # PCA-transformed likelihood
    if bdt_method_name == "LikelihoodPCA":
        factory.BookMethod( dataloader, TMVA.Types.kLikelihood, "LikelihoodPCA",
                            "!H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmooth=5:NAvEvtPerBin=50:VarTransform=PCA" )

    # Use a kernel density estimator to approximate the PDFs
    if bdt_method_name == "LikelihoodKDE":
        factory.BookMethod( dataloader, TMVA.Types.kLikelihood, "LikelihoodKDE",
                            "!H:!V:!TransformOutput:PDFInterpol=KDE:KDEtype=Gauss:KDEiter=Adaptive:KDEFineFactor=0.3:KDEborder=None:NAvEvtPerBin=50" )

    # Use a variable-dependent mix of splines and kernel density estimator
    if bdt_method_name == "LikelihoodMIX":
        factory.BookMethod( dataloader, TMVA.Types.kLikelihood, "LikelihoodMIX",
                            "!H:!V:!TransformOutput:PDFInterpolSig[0]=KDE:PDFInterpolBkg[0]=KDE:PDFInterpolSig[1]=KDE:PDFInterpolBkg[1]=KDE:PDFInterpolSig[2]=Spline2:PDFInterpolBkg[2]=Spline2:PDFInterpolSig[3]=Spline2:PDFInterpolBkg[3]=Spline2:KDEtype=Gauss:KDEiter=Nonadaptive:KDEborder=None:NAvEvtPerBin=50" )

    # Test the multi-dimensional probability density estimator
    # here are the options strings for the MinMax and RMS methods, respectively:
    #      "!H:!V:VolumeRangeMode=MinMax:DeltaFrac=0.2:KernelEstimator=Gauss:GaussSigma=0.3" );
    #      "!H:!V:VolumeRangeMode=RMS:DeltaFrac=3:KernelEstimator=Gauss:GaussSigma=0.3" );
    if bdt_method_name == "PDERS":
        factory.BookMethod( dataloader, TMVA.Types.kPDERS, "PDERS",
                            "!H:!V:NormTree=T:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600" )

    if bdt_method_name == "PDERSD":
        factory.BookMethod( dataloader, TMVA.Types.kPDERS, "PDERSD",
                            "!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=Decorrelate" )

    if bdt_method_name == "PDERSPCA":
        factory.BookMethod( dataloader, TMVA.Types.kPDERS, "PDERSPCA",
                             "!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=PCA" )

   # Multi-dimensional likelihood estimator using self-adapting phase-space binning
    if bdt_method_name == "PDEFoam":
        factory.BookMethod( dataloader, TMVA.Types.kPDEFoam, "PDEFoam",
                            "!H:!V:SigBgSeparate=F:TailCut=0.001:VolFrac=0.0666:nActiveCells=500:nSampl=2000:nBin=5:Nmin=100:Kernel=None:Compress=T" )

    if bdt_method_name == "PDEFoamBoost":
        factory.BookMethod( dataloader, TMVA.Types.kPDEFoam, "PDEFoamBoost",
                            "!H:!V:Boost_Num=30:Boost_Transform=linear:SigBgSeparate=F:MaxDepth=4:UseYesNoCell=T:DTLogic=MisClassificationError:FillFoamWithOrigWeights=F:TailCut=0:nActiveCells=500:nBin=20:Nmin=400:Kernel=None:Compress=T" )

    # K-Nearest Neighbour classifier (KNN)
    if bdt_method_name == "KNN":
        factory.BookMethod( dataloader, TMVA.Types.kKNN, "KNN",
                            "H:nkNN=20:ScaleFrac=0.8:SigmaFact=1.0:Kernel=Gaus:UseKernel=F:UseWeight=T:!Trim" )

    # H-Matrix (chi2-squared) method
    if bdt_method_name == "HMatrix":
        factory.BookMethod( dataloader, TMVA.Types.kHMatrix, "HMatrix", "!H:!V:VarTransform=None" )

    # Linear discriminant (same as Fisher discriminant)
    if bdt_method_name == "LD":
        factory.BookMethod( dataloader, TMVA.Types.kLD, "LD", "H:!V:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" )

    # Fisher discriminant (same as LD)
    if bdt_method_name == "Fisher":
        factory.BookMethod( dataloader, TMVA.Types.kFisher, "Fisher", "H:!V:Fisher:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" )

    # Fisher with Gauss-transformed input variables
    if bdt_method_name == "FisherG":
        factory.BookMethod( dataloader, TMVA.Types.kFisher, "FisherG", "H:!V:VarTransform=Gauss" )

    # Composite classifier: ensemble (tree) of boosted Fisher classifiers
    if bdt_method_name == "BoostedFisher":
        factory.BookMethod( dataloader, TMVA.Types.kFisher, "BoostedFisher",
                            "H:!V:Boost_Num=20:Boost_Transform=log:Boost_Type=AdaBoost:Boost_AdaBoostBeta=0.2:!Boost_DetailedMonitoring" )

    # Function discrimination analysis (FDA) -- test of various fitters - the recommended one is Minuit (or GA or SA)
    if bdt_method_name == "FDA_MC":
        factory.BookMethod( dataloader, TMVA.Types.kFDA, "FDA_MC",
                            "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1)(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MC:SampleSize=100000:Sigma=0.1" );

    if bdt_method_name == "FDA_GA":
        factory.BookMethod( dataloader, TMVA.Types.kFDA, "FDA_GA",
                            "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1)(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=GA:PopSize=300:Cycles=3:Steps=20:Trim=True:SaveBestGen=1" );

    if bdt_method_name == "FDA_SA":
        factory.BookMethod( dataloader, TMVA.Types.kFDA, "FDA_SA",
                            "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1)(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=SA:MaxCalls=15000:KernelTemp=IncAdaptive:InitialTemp=1e+6:MinTemp=1e-6:Eps=1e-10:UseDefaultScale" );

    if bdt_method_name == "FDA_MT":
        factory.BookMethod( dataloader, TMVA.Types.kFDA, "FDA_MT",
                            "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1)(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=2:UseImprove:UseMinos:SetBatch" );

    if bdt_method_name == "FDA_GAMT":
        factory.BookMethod( dataloader, TMVA.Types.kFDA, "FDA_GAMT",
                            "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1)(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=GA:Converger=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=0:!UseImprove:!UseMinos:SetBatch:Cycles=1:PopSize=5:Steps=5:Trim" );

    if bdt_method_name == "FDA_MCMT":
        factory.BookMethod( dataloader, TMVA.Types.kFDA, "FDA_MCMT",
                            "H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1)(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MC:Converger=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=0:!UseImprove:!UseMinos:SetBatch:SampleSize=20" );

    # TMVA ANN: MLP (recommended ANN) -- all ANNs in TMVA are Multilayer Perceptrons
    if bdt_method_name == "MLP":
        factory.BookMethod( dataloader, TMVA.Types.kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:!UseRegulator:Sampling=0.5:SamplingEpoch=0.8:ConvergenceTests=11:ConvergenceImprove=1e-6" )

    if bdt_method_name == "MLPBFGS":
        factory.BookMethod( dataloader, TMVA.Types.kMLP, "MLPBFGS", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:!UseRegulator" )

    if bdt_method_name == "MLPBNN":
        # factory.BookMethod( TMVA.Types.kMLP, "MLPBNN", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:UseRegulator" ) # BFGS training with bayesian regulators
        factory.BookMethod( dataloader, TMVA.Types.kMLP, "MLPBNN", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:TrainingMethod=BP:UseRegulator" )

    # Multi-architecture DNN implementation.
    if bdt_method_name == "DNN":
        # General layout.
        layoutString = "Layout=TANH|128,TANH|128,TANH|128,LINEAR"

        # Training strategies.
        training0 = "LearningRate=1e-1,Momentum=0.9,Repetitions=1,"\
                                "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"\
                                "WeightDecay=1e-4,Regularization=L2,"\
                                "DropConfig=0.0+0.5+0.5+0.5, Multithreading=True"
        training1 = "LearningRate=1e-2,Momentum=0.9,Repetitions=1,"\
                                "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"\
                                "WeightDecay=1e-4,Regularization=L2,"\
                                "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True"
        training2 = "LearningRate=1e-3,Momentum=0.0,Repetitions=1,"\
                                "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"\
                                "WeightDecay=1e-4,Regularization=L2,"\
                                "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True"
        trainingStrategyString = "TrainingStrategy="
        trainingStrategyString += training0 + "|" + training1 + "|" + training2

        # General Options.
        dnnOptions = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:"\
                                  "WeightInitialization=XAVIERUNIFORM"
        dnnOptions.Append (":"); dnnOptions.Append (layoutString)
        dnnOptions.Append (":"); dnnOptions.Append (trainingStrategyString)

        # Standard implementation, no dependencies.
        stdOptions = dnnOptions + ":Architecture=STANDARD"
        factory.BookMethod(dataloader, TMVA.Types.kDNN, "DNN", stdOptions)

        # Cuda implementation.
        if bdt_method_name == "DNN_GPU":
            gpuOptions = dnnOptions + ":Architecture=GPU"
            factory.BookMethod(dataloader, TMVA.Types.kDNN, "DNN GPU", gpuOptions)

        # Multi-core CPU implementation.
        if bdt_method_name == "DNN_CPU":
            cpuOptions = dnnOptions + ":Architecture=CPU"
            factory.BookMethod(dataloader, TMVA.Types.kDNN, "DNN CPU", cpuOptions)

    # CF(Clermont-Ferrand)ANN
    if bdt_method_name == "CFMlpANN":
        factory.BookMethod( dataloader, TMVA.Types.kCFMlpANN, "CFMlpANN", "!H:!V:NCycles=2000:HiddenLayers=N+1,N"  ) # n_cycles:#nodes:#nodes:...

    # Tmlp(Root)ANN
    if bdt_method_name == "TMlpANN":
        factory.BookMethod( dataloader, TMVA.Types.kTMlpANN, "TMlpANN", "!H:!V:NCycles=200:HiddenLayers=N+1,N:LearningMethod=BFGS:ValidationFraction=0.3"  ) # n_cycles:#nodes:#nodes:...

    # Support Vector Machine
    if bdt_method_name == "SVM":
        factory.BookMethod( dataloader, TMVA.Types.kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm" )

    # Boosted Decision Trees
    if bdt_method_name == "BDTG":
        factory.BookMethod( dataloader, TMVA.Types.kBDT, "BDTG",
                            "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=3")
    # Boosted Decision Trees
    if bdt_method_name == "BDTG3":
        factory.BookMethod( dataloader, TMVA.Types.kBDT, "BDTG3",
                                    "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.30:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=6")

    if bdt_method_name == "BDT":
        factory.BookMethod( dataloader, TMVA.Types.kBDT, "BDT",
                           "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" )

    if bdt_method_name == "BDTB":
        factory.BookMethod( dataloader, TMVA.Types.kBDT, "BDTB",
                           "!H:!V:NTrees=400:BoostType=Bagging:SeparationType=GiniIndex:nCuts=20" )

    if bdt_method_name == "BDTD":
        factory.BookMethod( dataloader, TMVA.Types.kBDT, "BDTD",
                           "!H:!V:NTrees=400:MinNodeSize=5%:MaxDepth=3:BoostType=AdaBoost:SeparationType=GiniIndex:nCuts=20:VarTransform=Decorrelate" )

    if bdt_method_name == "BDTF":  # Allow Using Fisher discriminant in node splitting for (strong) linearly correlated variables
        factory.BookMethod( dataloader, TMVA.Types.kBDT, "BDTF",
                                    "!H:!V:NTrees=50:MinNodeSize=2.5%:UseFisherCuts:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20")

    # RuleFit -- TMVA implementation of Friedman's method
    if bdt_method_name == "RuleFit":
        factory.BookMethod( dataloader, TMVA.Types.kRuleFit, "RuleFit",
                            "H:!V:RuleFitModule=RFTMVA:Model=ModRuleLinear:MinImp=0.001:RuleMinDist=0.001:NTrees=20:fEventsMin=0.01:fEventsMax=0.5:GDTau=-1.0:GDTauPrec=0.01:GDStep=0.01:GDNSteps=10000:GDErrScale=1.02" )

    # --------------------------------------------------------------------------------------------------

    # ---- Now you can tell the factory to train, test, and evaluate the MVAs.

    # Train MVAs
    factory.TrainAllMethods()

    # Test MVAs
    factory.TestAllMethods()

    # Evaluate MVAs
    factory.EvaluateAllMethods()

    # Save the output.
    outputFile.Close()

    print("=== wrote root file %s\n" % output_file)
    print("=== TMVAClassification is done!\n")

    # open the GUI for the result macros
    # gROOT.ProcessLine( "TMVAGui(\"%s\")" % output_file )

    # keep the ROOT thread running
    # gApplication.Run()


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    train_tmva(**vars(args))

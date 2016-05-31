from __future__ import division
__author__ = 'anitavalmarska'

from SD_learner_classifier import *
from math import sqrt
from math import log
from Beam_SD import *
from Apriori_SD import *
from CN2_SD import *
import os
import datetime
from os import walk
import Orange
import orange
import xlwt
from joblib import Parallel, delayed
import multiprocessing
#from numpy import array

###########################################################################################################
###########################################################################################################

class DoubleLoopSubgroupEvaluation():
    settingsList = [ "nFolds", "stdevBtnStatus"]

    eval_measures = [ ('Average coverage',         'COV', 'Coverage'),
                      ('Target support',           'SUP', 'Support'),
                      ('Average ruleset size',     'SIZE', 'Size'),
                      ('Average complexity',       'COMPLEX', 'Complexity'),
                      ('Average rule significance','SIG', 'Significance'),
                      ('Average rule unusualness', 'WRACC', 'Unusualness'),
                      ('Classification accuracy',  'CA', 'Classification accuracy'),
                      ('Area under ROC',           'AUC', 'AUC') ]

    g_values = [1,2,5,10,20,100]
    beam_width_values = [10,20,50,100]
    selection_beam_values = [5,10,20,30,50]
    max_confidence = [0.70, 0.80, 0.85, 0.90]
    #SD_weights = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    SD_weights = [0.1, 0.2, 0.5, 0.8, 0.85, 0.9, 0.95, 0.97]
    SD_algorithm_parameters = []
    for g in g_values:
        for w in beam_width_values:
            t = (g,w)
            SD_algorithm_parameters.append(t)

    """
    DB_weights_parameters = []
    #type = "geometric"

    if type == "harmonic":
        DB_weights_parameters = selection_beam_values
    elif type == "geometric":
        for w in SD_weights:
            for sb in selection_beam_values:
                t = (w,sb)
                DB_weights_parameters.append(t)
    """

    min_support = 0.01

    possible_learners = ['SD', 'APRIORI-SD', 'CN2-SD', 'DB-ILL', 'DB-IPP', 'DB-IMM', 'DB-IGG', 'DB-GG','DB-WRACC', 'DB-ILL-weights', 'DB-IPP-weights', 'DB-IMM-weights', 'DB-GG-weights', 'DB-IGG-weights', 'DB-WRACC-weights']

    def __init__(self):

        # Settings  =======================================================================================
        self.nFolds = 10                # cross validation folds
        self.usedMeasure = [1]*len(self.eval_measures)

        self.data = None                # input data set
        self.learndata = None           # separate learn data set
        self.testdata = None            # separate test data set
        self.learners = []              # set of learners (input)
        self.currentTargetClass = None  # current selection of target class
        self.builtClassifiers = []      # self.builtClassifiers => [ ( learner_id, [temp = list_of_classifiers] ) ]
        self.scores = dict()            # to be displayed in the table
        self.stdev = dict()

        self.selectedValue = []
        self.targetValues =[]
        self.learners = []
        self.name = ""
        self.times = dict()

        self.parametersUsed = dict()

    # change contents of targetValueList and prints data input properties
    def setData(self, data):

        if data and data.domain.hasDiscreteAttributes():
            # save new data
            self.data = data
            self.scores = dict()
            self.currentTargetClass = None

        # when data on input
        if self.data:
            # create learning set and testing set
            #self.learndata, self.testdata = self.sampleData(self.data, self.nFolds)

            self.targetValues = self.data.domain.classVar.values
            self.currentTargetClass = self.targetValues[0]
            self.selectedValue = [self.currentTargetClass]

        # when no data on input
        else:
            print "Please, provide data for evaluation."

    def set_number_of_rules(self, nr = 0):
        self.number_of_rules = nr

    # devide input data for cross validation
    def sampleData(self, data, nFolds):
        train = []
        test = []

        self.trainSets = {}
        self.evaluationSets = {}

        indeces = orange.MakeRandomIndicesCV(data, nFolds, randseed = False, \
                                             stratified = orange.MakeRandomIndices.StratifiedIfPossible)
        for fold in range(nFolds):
            train.append(data.select(indeces, fold, negate = 1))
            test.append(data.select(indeces, fold))
            learnSet, evaluationSet = self.splitTrainData(train[fold])

            self.trainSets[fold] = learnSet
            self.evaluationSets[fold] = evaluationSet

        return train, test

    def setName(self,name):
        self.name = name

    def setType(self,type):
        self.type = type
        self.DB_weights_parameters = []
        #type = "geometric"
        tempSDParameters = []

        if type == "harmonic":
            self.DB_weights_parameters = self.selection_beam_values
        elif type == "geometric":
            for w in self.SD_weights:
                for sb in self.selection_beam_values:
                    t = (w,sb)
                    self.DB_weights_parameters.append(t)

            for w in self.SD_weights:
                for comb in self.SD_algorithm_parameters:
                    temp = comb.append(w)
                    tempSDParameters.append(temp)

            self.SD_algorithm_parameters = tempSDParameters
        else:
            self.DB_weights_parameters = self.selection_beam_values

    def setLearners(self,learners):
        self.learners = learners

    def startDoubleEvaluation(self):
        if not self.data:
            print "Please, provide the data for evaluation."
        elif not len(self.learners):
            print "Please, provide the SD algorithms you want to evaluate. "

        else:
            self.doubleEvaluation()

    def doubleEvaluation(self):
        #perform double loop evaluation for each of the learners
        for learner in self.learners:
            if learner not in self.possible_learners:
                print "You can choose from one of the following learners:"
                print self.possible_learners
                continue

            bestResultsArray = {}
            print "New learner is about to be processed: ", datetime.datetime.now()
            print self.name, "\t", learner
            self.parametersUsed[learner] = []
            self.times[learner]={}
            self.times[learner]["start"] = datetime.datetime.now()
            for i in range(self.nFolds):
                trainData = self.learndata[i]
                testData = self.testdata[i]
                learnSet = self.trainSets[i]; evaluationSet = self.evaluationSets[i]
                parameters = self.getParameterList(learner=learner)
                j=0
                p_results = {}
                #print "Learn data: ", len(learnSet)
                #print "Evaluation set: ", len(evaluationSet)

                for parameter in parameters:
                    lrnr, model = self.buildLearner(learner=learner,ps=parameter,data=learnSet)
                    results = self.score(learner=lrnr, ruleSet=model,testData=evaluationSet)
                    p_results[j] = results
                    j+=1

                #best parameters are those which return the highest aggregate unusualness.
                # bestParameters_indx is the index of the best parameter
                bestParameters_indx = self.selectBestParameters(results=p_results)
                bestParameters = parameters[bestParameters_indx]

                self.parametersUsed[learner].append(bestParameters)

                lrnr, model = self.buildLearner(learner=learner,ps=bestParameters,data=Orange.data.Table(trainData))
                results = self.score(learner=lrnr,ruleSet=model,testData=Orange.data.Table(testData))
                bestResultsArray[i] = results

            averageResults = self.averageResults(learnerResults=bestResultsArray)
            self.storeResults(learner=learner,averageResults=averageResults)
            std_dev = self.calculateStandardDeviation(learner=learner,learnerResults=bestResultsArray)
            self.storeStd(learner=learner,std_dev=std_dev)
            self.times[learner]["end"] = datetime.datetime.now()

    def buildLearner(self,learner,ps,data):
        #build a learner, given its name, parameters and learning data
        num_rules = self.number_of_rules
        ruleSet = []
        if learner == "SD":
            lrnr = SD_learner(algorithm = 'SD', max_rules = num_rules, minSupport = self.min_support,\
                              beamWidth = ps[1], g = ps[0])

        elif learner == "APRIORI-SD":
            lrnr = SD_learner(algorithm = 'Apriori-SD', max_rules = num_rules, \
                              minSupport = self.min_support, minConfidence = ps, k=100)

        elif learner == "CN2-SD":
            lrnr = SD_learner(algorithm = 'CN2-SD', max_rules = num_rules, k = ps)

        elif learner == "DB-ILL":
            lrnr = SD_learner(algorithm = 'DoubleBeam', max_rules = num_rules, minSupport = self.min_support,\
                              beamWidth = ps, refinement_heuristics = "Inverted Laplace", selection_heuristics = "Laplace")

        elif learner == "DB-IPP":
            lrnr = SD_learner(algorithm = 'DoubleBeam', max_rules = num_rules, minSupport = self.min_support,\
                              beamWidth = ps, refinement_heuristics = "Inverted precision", selection_heuristics = "Precision")

        elif learner == "DB-IMM":
            lrnr = SD_learner(algorithm = 'DoubleBeam', max_rules = num_rules, minSupport = self.min_support,\
                              beamWidth = ps, refinement_heuristics = "Inverted m", selection_heuristics = "m-est")

        elif learner == "DB-IGG":
            lrnr = SD_learner(algorithm = 'DoubleBeam', max_rules = num_rules, minSupport = self.min_support,\
                              beamWidth = ps[1], g= ps[0], refinement_heuristics = "Inverted g", selection_heuristics = "g")

        elif learner == "DB-GG":
            lrnr = SD_learner(algorithm = 'DoubleBeam', max_rules = num_rules, minSupport = self.min_support,\
                              beamWidth = ps[1], g= ps[0], refinement_heuristics = "g", selection_heuristics = "g")

        elif learner == "DB-WRACC":
            lrnr = SD_learner(algorithm = 'DoubleBeam', max_rules = num_rules, minSupport = self.min_support,\
                              beamWidth = ps, refinement_heuristics = "WRACC", selection_heuristics = "WRACC")

        elif learner == "DB-ILL-weights":
            if self.type == "harmonic":
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps, refinement_heuristics="Inverted Laplace", selection_heuristics="Laplace")
            elif self.type == "geometric":
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps[1], refinement_heuristics="Inverted Laplace", selection_heuristics="Laplace",
                                  db_type=self.type, weight_factor=ps[0])

            else:
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps, refinement_heuristics="Inverted Laplace", selection_heuristics="Laplace",
                                  db_type=self.type)
            #lrnr = SD_learner(algorithm='DoubleBeam', max_rules=num_rules, minSupport=self.min_support, \
            #                  beamWidth=ps, refinement_heuristics="Inverted Laplace", selection_heuristics="Laplace")

        elif learner == "DB-IPP-weights":
            if self.type == "harmonic":
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps, refinement_heuristics="Inverted precision",
                                  selection_heuristics="Precision")
            elif self.type == "geometric":
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps[1], refinement_heuristics="Inverted precision",
                                  selection_heuristics="Precision",
                                  db_type=self.type, weight_factor=ps[0])

            else:
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps, refinement_heuristics="Inverted precision",
                                  selection_heuristics="Precision",
                                  db_type=self.type)
            #lrnr = SD_learner(algorithm='DoubleBeam', max_rules=num_rules, minSupport=self.min_support, \
            #                  beamWidth=ps, refinement_heuristics="Inverted precision",
            #                  selection_heuristics="Precision")

        elif learner == "DB-IMM-weights":
            if self.type == "harmonic":
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps, refinement_heuristics="Inverted m",
                                  selection_heuristics="m-est")
            elif self.type == "geometric":
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps[1], refinement_heuristics="Inverted m",
                                  selection_heuristics="m-est",
                                  db_type=self.type, weight_factor=ps[0])

            else:
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps, refinement_heuristics="Inverted m",
                                  selection_heuristics="m-est",
                                  db_type=self.type)
            #lrnr = SD_learner(algorithm='DoubleBeam', max_rules=num_rules, minSupport=self.min_support, \
            #                  beamWidth=ps, refinement_heuristics="Inverted m", selection_heuristics="m-est")

        elif learner == "DB-IGG-weights":
            if self.type == "harmonic":
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps[1], g=ps[0], refinement_heuristics="Inverted g",
                                  selection_heuristics="g")
            elif self.type == "geometric":
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  g=ps[0], beamWidth=ps[1], refinement_heuristics="Inverted g", selection_heuristics="g",
                                  db_type=self.type, weight_factor=ps[2])

            else:
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps[1], g=ps[0], refinement_heuristics="Inverted g",
                                  selection_heuristics="g",
                                  db_type=self.type)
            #lrnr = SD_learner(algorithm='DoubleBeam', max_rules=num_rules, minSupport=self.min_support, \
            #                  beamWidth=ps[1], g=ps[0], refinement_heuristics="Inverted g", selection_heuristics="g")

        elif learner == "DB-GG-weights":
            if self.type == "harmonic":
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  g=ps[0], beamWidth=ps[1], refinement_heuristics="g",
                                  selection_heuristics="g")
            elif self.type == "geometric":
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps[1], refinement_heuristics="g", g=ps[0],
                                  selection_heuristics="g",
                                  db_type=self.type, weight_factor=ps[2])

            else:
                lrnr = SD_learner(algorithm='DoubleBeam-weights', max_rules=num_rules, minSupport=self.min_support, \
                                  beamWidth=ps[1], refinement_heuristics="g", g = ps[0],
                                  selection_heuristics="g", db_type=self.type)
            #lrnr = SD_learner(algorithm='DoubleBeam', max_rules=num_rules, minSupport=self.min_support, \
            #                  beamWidth=ps[1], g=ps[0], refinement_heuristics="g", selection_heuristics="g")

        elif learner == "DB-WRACC-weights":
            if self.type == "harmonic":
                lrnr = SD_learner(algorithm = 'DoubleBeam-weights', max_rules = num_rules, minSupport = self.min_support,\
                              beamWidth = ps, refinement_heuristics = "WRACC", selection_heuristics = "WRACC")
            elif self.type == "geometric":
                lrnr = SD_learner(algorithm = 'DoubleBeam-weights', max_rules = num_rules, minSupport = self.min_support,\
                              beamWidth = ps[1], refinement_heuristics = "WRACC", selection_heuristics = "WRACC", db_type = self.type, weight_factor = ps[0])

            else:
                lrnr = SD_learner(algorithm = 'DoubleBeam-weights', max_rules = num_rules, minSupport = self.min_support,\
                              beamWidth = ps, refinement_heuristics = "WRACC", selection_heuristics = "WRACC", db_type = self.type)

        else:
            print "Incorrect learner. Rule set is empty."
            return ruleSet

        ruleSet = lrnr(data)
        return lrnr, ruleSet

    def getParameterList(self,learner):
        if learner == "SD" or learner == "DB-IGG" or learner == "DB-GG":
            return self.SD_algorithm_parameters
        elif learner == "CN2-SD":
            return self.beam_width_values
        elif learner == "APRIORI-SD":
            return self.max_confidence
        elif learner == "DB-ILL" or learner == "DB-IPP" or learner == "DB-IMM" or learner == "DB-WRACC":
            return self.selection_beam_values
        elif learner in ['DB-ILL-weights', 'DB-IPP-weights', 'DB-IMM-weights', 'DB-GG-weights', 'DB-IGG-weights', 'DB-WRACC-weights']:
            return self.DB_weights_parameters
        else:
            print "Incorrect learner. Empty set of parameters."
            return []

    def selectBestParameters(self,results):
        #this function returns the index of the parameters combination with maximum aggregate unusualness
        maxUnusualness = -10
        bestParameterIndx = 0
        for i in range(len(results)):
            tempUnusualness = 0
            for c in self.targetValues:
                tempUnusualness += results[i][c]["WRACC"]
            if tempUnusualness > maxUnusualness:
                maxUnusualness = tempUnusualness
                bestParameterIndx = i
        return bestParameterIndx

    def averageResults(self, learnerResults):
        #this function calculates the average values for each of the evaluation measures, for each class
        average_results = {}
        for c in self.targetValues:
            coverage = 0; support = 0; size = 0; complexity = 0; significance = 0; unusualness = 0; ca = 0; auc = 0;
            nFolds = len(learnerResults)
            for i in range(nFolds):
                coverage += learnerResults[i][c]['COV']
                support += learnerResults[i][c]['SUP']
                size += learnerResults[i][c]['SIZE']
                complexity += learnerResults[i][c]['COMPLEX']
                significance += learnerResults[i][c]['SIG']
                unusualness += learnerResults[i][c]['WRACC']
                ca += learnerResults[i][c]['CA']
                auc += learnerResults[i][c]['AUC']

            average_coverage = coverage/nFolds
            average_support = support/nFolds
            average_size = size/nFolds
            average_complexity = complexity/nFolds
            average_significance = significance/nFolds
            average_WRACC = unusualness/nFolds
            average_ca = ca/nFolds
            average_auc = auc/nFolds

            average_results[c] = {"COV":average_coverage, "SUP":average_support, "SIZE":average_size,\
                                  "COMPLEX":average_complexity, "SIG":average_significance, "WRACC":average_WRACC, \
                                  "CA":average_ca, "AUC":average_auc}

        #print "Average results: ", average_results
        return average_results

    def calculateStandardDeviation(self,learner,learnerResults):
        std_results = {}
        #nFolds = nFolds = len(learnerResults)
        for c in self.targetValues:
            coverage = 0; support = 0; size = 0; complexity = 0; significance = 0; unusualness = 0; ca = 0; auc = 0;
            nFolds = len(learnerResults)
            n = nFolds-1
            for i in range(nFolds):
                coverage += (learnerResults[i][c]['COV']-self.scores[learner][c]['COV'])**2
                support += (learnerResults[i][c]['SUP']-self.scores[learner][c]['SUP'])**2
                size += (learnerResults[i][c]['SIZE']-self.scores[learner][c]['SIZE'])**2
                complexity += (learnerResults[i][c]['COMPLEX']-self.scores[learner][c]['COMPLEX'])**2
                significance += (learnerResults[i][c]['SIG']-self.scores[learner][c]['SIG'])**2
                unusualness += (learnerResults[i][c]['WRACC']-self.scores[learner][c]['WRACC'])**2
                ca += (learnerResults[i][c]['CA']-self.scores[learner][c]['CA'])**2
                auc += (learnerResults[i][c]['AUC']-self.scores[learner][c]['AUC'])**2

            if nFolds<=1:
                std_results[c] = {"COV":0.0, "SUP":0.0, "SIZE":0.0, "COMPLEX":0.0, "SIG":0.0, "WRACC":0.0, "CA":0.0, "AUC":0.0}

            else:
                std_coverage = sqrt(coverage/n)
                std_support = sqrt(support/n)
                std_size = sqrt(size/n)
                std_complexity = sqrt(complexity/n)
                std_significance = sqrt(significance/n)
                std_WRACC = sqrt(unusualness/n)
                std_ca = sqrt(ca/n)
                std_auc = sqrt(auc/n)

            std_results[c] = {"COV":std_coverage, "SUP":std_support, "SIZE":std_size, "COMPLEX":std_complexity,\
                              "SIG":std_significance, "WRACC":std_WRACC, "CA":std_ca, "AUC":std_auc}

        return std_results

    def splitTrainData(self,data):
        #data is split into train and test data in ratio 2:1
        nFolds = 3
        train = []
        test = []
        indeces = orange.MakeRandomIndicesCV(data, nFolds, randseed = False,\
                                             stratified = orange.MakeRandomIndices.StratifiedIfPossible)
        #we take only the first combination
        train.append(data.select(indeces, 0, negate = 1))
        test.append(data.select(indeces, 0))
        return Orange.data.Table(train), Orange.data.Table(test)

    def storeResults(self,learner,averageResults):
        self.scores[learner] = averageResults

    def storeStd(self,learner,std_dev):
        self.stdev[learner]=std_dev

    # calculating evaluating measures
    def score(self, learner, ruleSet, testData):

        def calcHull(subgroups, Y, X, A, B):
            #inicialization
            C = (-1,-1)    # best new point point
            y = -1         # best distance

            # calculate best new point
            if (B[0]-A[0]) != 0:
                k = (B[1]-A[1]) / (B[0]-A[0])  # coefficient of the line between A and B
                for i in range(len(Y)):        # check every point
                    yn = Y[i] -( k * ( X[i] - A[0] ) + A[1])   # vertical distance between point i and line AB
                    if yn>0 and yn > y:        # if new distance is the greatest so far
                        C = (X[i], Y[i])       # the new point is the best so far
                        y = yn

            # if new point on the hull was found
            if C != (-1,-1):
                # recursivey call this function on the LEFT side of the point
                Xl =[]
                Yl =[]
                for i in range(len(X)):
                    if X[i]>A[0] and X[i]<C[0]:
                        Xl.append(X[i])
                        Yl.append(Y[i])
                calcHull(subgroups, Yl, Xl, A,C)  # recursive call

                subgroups.hullTPR.append(C[1])
                subgroups.hullFPR.append(C[0])

                # recursivey call this function on the RIGHT side of the point
                Xu =[]
                Yu =[]
                for i in range(len(X)):
                    if X[i]>C[0] and X[i]<B[0]:
                        Xu.append(X[i])
                        Yu.append(Y[i])
                calcHull(subgroups, Yu, Xu, C,B)  # recursive call

        def calcAUC(X,Y):
            area = 0.0
            for i in range(len(X)-1):
                x = X[i+1]-X[i]
                y1 = Y[i]
                y2 = Y[i+1]
                trapez = x* (y1+y2)/2
                area = area + trapez
            return area

        # return the size of induced ruleset for one target class value (without default rule)
        def calculate(learner, ruleSet, testData):
            size = 0.0
            complexity = 0.0
            covarage = 0.0
            support = []
            significance = 0.0
            unusualness = 0.0
            accuracy = 0.0
            X = []
            Y = []
            learner.hullTPR = [0]
            learner.hullFPR = [0]
            TP_table = set()
            TN_table = set()

            if ruleset:
                N = len(testData)
                temp = orange.getClassDistribution(testData)
                Poz = temp[ruleSet[0].targetClass]
                Neg = 0
                tempN = filter(lambda t: t != ruleSet[0].targetClass, temp.keys())
                for t in tempN:
                    Neg += temp[t]

                # correct when varibale is 0
                if Poz == 0:
                    Poz = 1e-15
                if Neg == 0:
                    Neg = 1e-15

                for rule in ruleset:
                    # if we don't have dafault rule
                    if len(rule.filter.conditions):
                        TP = filter(lambda e: e.getclass() == rule.targetClass, rule.filter(testData))
                        FP = filter(lambda e: e.getclass() != rule.targetClass, rule.filter(testData))
                        TN = filter(lambda e: e.getclass() != rule.targetClass, rule.filter(testData, negate = 1))
                        nX = len(TP) + len(FP)
                        nY = int(orange.getClassDistribution(testData)[rule.targetClass])
                        nXY = len(TP)
                        cov = float(nX)/N

                        # correct when variable is 0
                        if nX == 0:
                            nX = 1e-15
                        if nY == 0:
                            nY = 1e-15
                        if nXY == 0:
                            nXY = 1e-15
                        if cov == 0:
                            cov = 1e-15


                        for e in TP:
                            support.append(e)
                            TP_table.add(e)
                        for e in TN:
                            TN_table.add(e)

                        covarage += cov
                        size += 1.0
                        complexity += rule.complexity
                        if (float(nXY) / (nY * cov)) > 1.0:
                            significance += 2.0 * nXY * log( float(nXY) / (nY * cov) )
                        unusualness += cov * (float(nXY)/nX - float(nY)/N)
                        accuracy += (float(len(TP)) + float(len(TN)))/N
                        X.append(float(len(FP))/Neg)
                        Y.append(float(len(TP))/Poz)

                calcHull(learner, Y, X , A=(0,0), B=(1,1))
                learner.hullTPR.append(1)
                learner.hullFPR.append(1)
                auc = calcAUC(learner.hullFPR, learner.hullTPR)

                #if size > 0:
                #    stdev_list[6].append(float(float(accuracy)/size))
                #else:
                #    stdev_list[6].append(float("infinity"))
                #stdev_list[7].append(calcAUC(learner.hullFPR, learner.hullTPR))


            #print len(ruleSet)
            if size > 0:
                    return float(size), float(complexity)/size, float(covarage)/size, \
                    float(len(set(support)))/Poz, float(significance)/size, float(unusualness)/size, \
                    float(float(accuracy)/size), float(auc)
            else:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        def calcStdev(list):
            if len(list):
                avg = sum(list)/len(list)
                sdsq = sum([(i - avg) ** 2 for i in list])
                return (sdsq / len(list)) ** .5
            return -1.0

################################################################################


        # add new variables which will hold results
        results = dict()
        for rulesClass in ruleSet.rulesClass:
            # set of induced rules for this target class value
            ruleset = rulesClass.rules.rules
            #print "Len ruleset: ", len(ruleset)
            cls = ruleset[0].targetClass.value

                # addting result in list
            siz, comp, cov, sup, sig, \
            unusual, acc, auc = calculate(learner, ruleset, testData)

            temp_results = dict()
            temp_results = {"SIZE":siz, "COMPLEX":comp, "COV":cov, "SUP":sup,"SIG":sig, "WRACC":unusual, "CA":acc, "AUC":auc}
            results[cls] = temp_results

        return results

    def writeResults(self,file_name):
        #current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/NEWEST_RESULTS/IMM_NO_POSTPROCESSING"+self.type
        #current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/NEWEST_RESULTS/WEIGHTS/"+self.type+"/postprocessing_"+str(self.number_of_rules)+r"/measures"
        current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/COMPARISON_RESULTS/STATE_OF_THE_ART/postprocessing_"+str(self.number_of_rules)+r"/measures"
        #current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/COMPARISON_RESULTS/DOUBLE_BEAM/postprocessing_"+str(self.number_of_rules)+r"/measures"


        if not os.path.exists(current_directory):
            os.makedirs(current_directory)
        excel_file = os.path.join(current_directory,file_name)

        workbook = xlwt.Workbook()
        sheets = {}

        decimal_style = xlwt.XFStyle()
        decimal_style.num_format_str = '0.000'

        target_class = {}

        n = len(self.targetValues)

        for i in range(n):
            sn = self.targetValues[i]
            sheet_name = sn.replace('\'','')
            sheets[i] = workbook.add_sheet(sheet_name)
            target_class[i]=self.targetValues[i]

        total_WRACC = {}; total_complexity = {}; total_coverage = {}; total_support = {};
        total_size = {}; total_significance = {}; total_ca = {}; total_auc = {};

        for learner in self.learners:
            total_WRACC[learner] = 0; total_complexity[learner] = 0; total_coverage[learner] = 0; total_support[learner] = 0;
            total_size[learner] = 0; total_significance[learner] = 0; total_ca[learner] = 0; total_auc[learner] = 0;

        if self.scores:
            for i in range(n):
                cls = target_class[i]

                sheets[i].write(0,0,"Measure")
                sheets[i].write(1,0,"WRACC")
                sheets[i].write(2,0,"Complexity")
                sheets[i].write(3,0,"Coverage")
                sheets[i].write(4,0,"Support")
                sheets[i].write(5,0,"Size")
                sheets[i].write(6,0,"Significance")
                sheets[i].write(7,0,"CA")
                sheets[i].write(8,0,"AUC")

                for j,learner in enumerate(self.learners):
                    sheets[i].write(0,j+1,learner)

                    WRACC = self.scores[learner][cls]["WRACC"];         total_WRACC[learner] += WRACC
                    complexity = self.scores[learner][cls]["COMPLEX"];  total_complexity[learner] += complexity
                    coverage = self.scores[learner][cls]["COV"];        total_coverage[learner] += coverage
                    support = self.scores[learner][cls]["SUP"];         total_support[learner] += support
                    size = self.scores[learner][cls]["SIZE"];           total_size[learner] += size
                    significance = self.scores[learner][cls]["SIG"];    total_significance[learner] += significance
                    ca = self.scores[learner][cls]["CA"];               total_ca[learner] += ca
                    auc = self.scores[learner][cls]["AUC"];             total_auc[learner] += auc

                    sheets[i].write(1,j+1,WRACC,decimal_style)
                    sheets[i].write(2,j+1,complexity,decimal_style)
                    sheets[i].write(3,j+1,coverage,decimal_style)
                    sheets[i].write(4,j+1,support,decimal_style)
                    sheets[i].write(5,j+1,size,decimal_style)
                    sheets[i].write(6,j+1,significance,decimal_style)
                    sheets[i].write(7,j+1,ca,decimal_style)
                    sheets[i].write(8,j+1,auc,decimal_style)


            average_sheet = workbook.add_sheet("Average results")
            average_sheet.write(0,0,"Measure")
            average_sheet.write(1,0,"WRACC")
            average_sheet.write(2,0,"Complexity")
            average_sheet.write(3,0,"Coverage")
            average_sheet.write(4,0,"Support")
            average_sheet.write(5,0,"Size")
            average_sheet.write(6,0,"Significance")
            average_sheet.write(7,0,"CA")
            average_sheet.write(8,0,"AUC")


            for j,learner in enumerate(self.learners):
                average_sheet.write(0,j+1,learner)
                average_sheet.write(1,j+1,total_WRACC[learner]/n,decimal_style)
                average_sheet.write(2,j+1,total_complexity[learner]/n,decimal_style)
                average_sheet.write(3,j+1,total_coverage[learner]/n,decimal_style)
                average_sheet.write(4,j+1,total_support[learner]/n,decimal_style)
                average_sheet.write(5,j+1,total_size[learner]/n,decimal_style)
                average_sheet.write(6,j+1,total_significance[learner]/n,decimal_style)
                average_sheet.write(7,j+1,total_ca[learner]/n,decimal_style)
                average_sheet.write(8,j+1,total_auc[learner]/n,decimal_style)

        workbook.save(excel_file)

    def writeStdev(self,file_name):
        #current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/results/weights/stdev/"+self.type
        #current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/NEWEST_RESULTS/WEIGHTS/"+self.type+"/postprocessing_"+str(self.number_of_rules)+r"/stdev"
        current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/COMPARISON_RESULTS/STATE_OF_THE_ART/postprocessing_"+str(self.number_of_rules)+r"/stdev"
        #current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/COMPARISON_RESULTS/DOUBLE_BEAM/postprocessing_"+str(self.number_of_rules)+r"/stdev"


        if not os.path.exists(current_directory):
            os.makedirs(current_directory)
        excel_file = os.path.join(current_directory,file_name)

        workbook = xlwt.Workbook()
        sheets = {}

        decimal_style = xlwt.XFStyle()
        decimal_style.num_format_str = '0.000'

        target_class = {}

        n = len(self.targetValues)

        for i in range(n):
            sn = self.targetValues[i]
            sheet_name = sn.replace('\'','')
            sheets[i] = workbook.add_sheet(sheet_name)
            target_class[i]=self.targetValues[i]

        if self.stdev:
            for i in range(n):
                cls = target_class[i]

                sheets[i].write(0,0,"Measure")
                sheets[i].write(1,0,"WRACC")
                sheets[i].write(2,0,"Complexity")
                sheets[i].write(3,0,"Coverage")
                sheets[i].write(4,0,"Support")
                sheets[i].write(5,0,"Size")
                sheets[i].write(6,0,"Significance")
                sheets[i].write(7,0,"CA")
                sheets[i].write(8,0,"AUC")

                for j,learner in enumerate(self.learners):
                    sheets[i].write(0,j+1,learner)

                    WRACC = self.stdev[learner][cls]["WRACC"];
                    complexity = self.stdev[learner][cls]["COMPLEX"];
                    coverage = self.stdev[learner][cls]["COV"];
                    support = self.stdev[learner][cls]["SUP"];
                    size = self.stdev[learner][cls]["SIZE"];
                    significance = self.stdev[learner][cls]["SIG"];
                    ca = self.stdev[learner][cls]["CA"];
                    auc = self.stdev[learner][cls]["AUC"];

                    sheets[i].write(1,j+1,WRACC,decimal_style)
                    sheets[i].write(2,j+1,complexity,decimal_style)
                    sheets[i].write(3,j+1,coverage,decimal_style)
                    sheets[i].write(4,j+1,support,decimal_style)
                    sheets[i].write(5,j+1,size,decimal_style)
                    sheets[i].write(6,j+1,significance,decimal_style)
                    sheets[i].write(7,j+1,ca,decimal_style)
                    sheets[i].write(8,j+1,auc,decimal_style)

        workbook.save(excel_file)

    def writeParametersUsed(self,file_name):
        #current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/results/weights/parameters/"+self.type

        #current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/NEWEST_RESULTS/WEIGHTS/"+self.type+"/postprocessing_"+str(self.number_of_rules)+r"/parameters"
        current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/COMPARISON_RESULTS/STATE_OF_THE_ART/postprocessing_" + str(self.number_of_rules) + r"/parameters"
        #current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/COMPARISON_RESULTS/DOUBLE_BEAM/postprocessing_" + str(self.number_of_rules) + r"/parameters"


        if not os.path.exists(current_directory):
            os.makedirs(current_directory)
        file_address = os.path.join(current_directory,file_name)

        f = open(file_address,'w')

        for learner in self.learners:
            f.write("Learner: "+learner)
            f.write("\n")
            for parameters in self.parametersUsed[learner]:
                f.write(str(parameters)+"\t")
            f.write("\n"+"#"*20)

        f.close()

    def writeExecutionTimes(self,file_name):
        #current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/results/weights/running_times/"+self.type
        current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/NEWEST_RESULTS/WEIGHTS/"+self.type+"/postprocessing_"+str(self.number_of_rules)+r"/running_times"
        current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/COMPARISON_RESULTS/DOUBLE_BEAM/postprocessing_" + str(self.number_of_rules) + r"/running_times"
        current_directory = os.path.dirname(
            os.path.realpath(__file__)) + r"/COMPARISON_RESULTS/STATE_OF_THE_ART/postprocessing_" + str(self.number_of_rules) + r"/running_times"
        if not os.path.exists(current_directory):
            os.makedirs(current_directory)
        file_address = os.path.join(current_directory,file_name)

        f = open(file_address,'w')

        for learner in self.learners:
            start = self.times[learner]["start"]
            end = self.times[learner]["end"]
            diff = end - start
            run_time_mins = divmod(diff.days*86400+diff.seconds,60)
            f.write(learner+"\t"+unicode(start)[:-7]+"\t"+unicode(end)[:-7]+"\t"+str(diff)+"\n")

        f.close()

    def get_parsed_data(self, dataset):
        mypath = os.path.dirname(os.path.realpath(__file__)) + r"/20_DATASETS_TAB"
        d_a = os.path.join(mypath, dataset+".tab")
        data = Orange.data.Table(d_a)
        self.setData(data)


        database_directory = os.path.dirname(os.path.realpath(__file__)) + r"/datasets_separated/"
        address_train = database_directory+dataset+r"/learn"
        address_test = database_directory+dataset+r"/test"

        self.learndata = dict(); self.trainSets = dict();
        self.testdata = dict(); self.evaluationSets = dict();

        for i in range(self.nFolds):
            train_a = os.path.join(address_train,str(i)+".tab")
            test_a = os.path.join(address_test, str(i) + ".tab")
            self.learndata[i] = Orange.data.Table(train_a)
            self.testdata[i] = Orange.data.Table(test_a)

            learnSet, evaluationSet = self.splitTrainData(self.learndata[i])

            self.trainSets[i] = learnSet
            self.evaluationSets[i] = evaluationSet

    def evaluate(self):
        # perform double loop evaluation for each of the learners
        for learner in self.learners:
            if learner not in self.possible_learners:
                print "You can choose from one of the following learners:"
                print self.possible_learners
                continue

            bestResultsArray = {}
            print "New learner is about to be processed: ", datetime.datetime.now()
            print self.name, "\t", learner
            self.parametersUsed[learner] = []
            self.times[learner] = {}
            self.times[learner]["start"] = datetime.datetime.now()
            for i in range(self.nFolds):
                trainData = self.learndata[i]
                testData = self.testdata[i]
                parameters = self.get_parameters(learner=learner)

                lrnr, model = self.buildLearner(learner=learner, ps=parameters, data=trainData)
                results = self.score(learner=lrnr, ruleSet=model, testData=testData)

                bestResultsArray[i] = results

            averageResults = self.averageResults(learnerResults=bestResultsArray)
            self.storeResults(learner=learner, averageResults=averageResults)
            std_dev = self.calculateStandardDeviation(learner=learner, learnerResults=bestResultsArray)
            self.storeStd(learner=learner, std_dev=std_dev)
            self.times[learner]["end"] = datetime.datetime.now()

    def get_parameters(self, learner=""):
        if learner == "SD":
            return [10,10]
            pass
        if learner == "APRIORI-SD":
            return 0.7
            pass
        if learner == "CN2-SD":
            return 10
            pass
        if learner == "DB-IPP-weights":
            return 5
            pass
        if learner == "DB-ILL":
            return 5
            pass
        if learner == "DB-WRACC":
            return 5
            pass


def call_SD_evaluation(dataset="",learner="",type="",nr=0):

    ow = DoubleLoopSubgroupEvaluation()
    ow.setName(dataset)
    ow.set_number_of_rules(nr)
    ow.setType(type)

    ow.get_parsed_data(dataset)
    ow.setLearners([learner])

    ow.startDoubleEvaluation()

    ow.writeResults(dataset+"_"+learner+".xls")
    ow.writeStdev(dataset+"_"+learner+".xls")
    ow.writeParametersUsed(dataset+"_"+learner+".txt")
    ow.writeExecutionTimes(dataset+"_"+learner+".txt")

def call_SD_evaluation_SET_PARAMETERS(dataset="",learner="",type="",nr=0):
    ow = DoubleLoopSubgroupEvaluation()
    ow.setName(dataset)
    ow.set_number_of_rules(nr)
    ow.setType(type)

    ow.get_parsed_data(dataset)
    ow.setLearners([learner])

    ow.evaluate()

    ow.writeResults(dataset + "_" + learner + ".xls")
    ow.writeStdev(dataset + "_" + learner + ".xls")
    ow.writeParametersUsed(dataset + "_" + learner + ".txt")
    ow.writeExecutionTimes(dataset + "_" + learner + ".txt")

#datasets = ['contact-lenses','futebol','iris','ionosphere', 'labor', 'mushroom','primary-tumor', 'soybean', 'tic-tac-toe','zoo']

datasets = ['contact-lenses']
learners = ['SD', 'CN2-SD', 'APRIORI-SD']
nrs = [0,5]


combinations = []

for dataset in datasets:
    for learner in learners:
        for nr in nrs:
            #for type in types:
            combination = [dataset, learner, "", nr]
            combinations.append(combination)


num_cores = multiprocessing.cpu_count()

#BEFORE RUNNING, PLEASE CHECK THE WRITING ADDRESSES
Parallel(n_jobs=num_cores)(
    delayed(call_SD_evaluation_SET_PARAMETERS)(combinations[k][0], combinations[k][1], combinations[k][2], combinations[k][3]) for k in range(len(combinations)))

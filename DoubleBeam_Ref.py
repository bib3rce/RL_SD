import Orange
import orange
import sys
import operator
from datetime import datetime
from SDRule import *

true = 1
false = 0

class DoubleBeam:
    def __init__(self, minSupport = 0.2, sb_width=10, rb_width = 10, g=1, refinement_heuristics = "Inverted Laplace", selection_heuristics = "Laplace", **kwds):
        self.minSupport = minSupport
        self.g = g
        self.CandidatesList = set()
        self.selectionBeamWidth = sb_width
        self.refinementBeamWidth = rb_width
        self.refinementCandidates = []
        self.selectionCandidates = []
        self.alredyRefinedRules = dict()
        self.refinement_heuristics = refinement_heuristics
        self.selection_heuristics = selection_heuristics
        self.alreadySelectedRules = set()

    def __call__(self, data, targetClass, num_of_rules ):
        self.alredyRefinedRules[targetClass] = set()
        if self.dataOK(data):  # Checks weather targetClass is discrete
            data_discretized = False
            # If any of the attributes are continuous, discretize them
            if data.domain.hasContinuousAttributes():
                original_data = data
                data_discretized = True
                new_domain = []
                discretize = orange.EntropyDiscretization(forceAttribute=True)
                for attribute in data.domain.attributes:
                    if attribute.varType == orange.VarTypes.Continuous:
                        d_attribute = discretize(attribute, data)
                        new_domain.append(d_attribute)
                    else:
                        new_domain.append(attribute)
                data = original_data.select(new_domain + [original_data.domain.classVar])

            self.data = data
            self.targetClass = targetClass

            #Initialize CanditatesList (all features)
            self.fillCandidatesList(data,targetClass)

            #Initialize RefinementBeam, consisting of refinementBeamWidth empty rules
            self.initializeRefinementBeam()
            #Initialize SelectionBeam, consisting of selectionBeamWidth empty rules
            self.initializeSelectionBeam()

            #update RefinementBeam
            self.updateRefinementBeam(self.refinementCandidates)
            #update SelectionBeam
            self.updateSelectionBeam(self.selectionCandidates)

            improvements = True
            refinement_improvements = True

            i=2
            max_steps=5

            while improvements:
                self.refinedRefinementBeam(targetClass)
                self.updateRefinementBeam(self.refinementCandidates)
                improvements = self.updateSelectionBeam(self.selectionCandidates)
                i=i+1

            beam = self.SelectionBeam
            if num_of_rules != 0:
                beam = self.ruleSubsetSelection(beam, num_of_rules, data)
                self.SelectionBeam = beam

            if data_discretized:
                 targetClassRule = SDRule(original_data, targetClass, conditions=[], g=self.g)
                 self.SelectionBeam = [rule.getUndiscretized(original_data) for rule in self.SelectionBeam]

            else:
                 targetClassRule = SDRule(data, targetClass, conditions=[], g =self.g)

            rules = SDRules(self.SelectionBeam, targetClassRule, "DoubleBeam-RL")
            return rules

    def fillCandidatesList(self, data, targetClass):
        #first initialize empty rule
        rule = SDRule(data=data, targetClass=targetClass, g=self.g, refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)
        newRefinementBeam = {}
        newSelectionBeam = {}

        self.alredyRefinedRules[targetClass].add(rule.orderedRuleToString())

        for attr in data.domain.attributes:
            value = attr.firstvalue()
            while(value):
                newRule = rule.cloneAndAddCondition(attr,value,rule,refinement_heuristics=self.refinement_heuristics,selection_heuristics=self.selection_heuristics)
                newRule.filterAndStore(rule)
                self.CandidatesList.add(newRule)
                newRefinementBeam[newRule] = newRule.refinement_quality
                newSelectionBeam[newRule] = newRule.selection_quality
                value = attr.nextvalue(value)

        no_candidates = len(self.CandidatesList)

        #sort the rules according to their refinement qualities
        sorted_newRefinementBeam = sorted(newRefinementBeam.items(), key=operator.itemgetter(1), reverse=True)
        self.refinementCandidates = [i[0] for i in sorted_newRefinementBeam]

        #sort the rules according to their selection quality
        sorted_newSelectionBeam = sorted(newSelectionBeam.items(), key=operator.itemgetter(1), reverse=True)
        l_sortedNewSelectionBeam = [i[0] for i in sorted_newSelectionBeam]
        self.sortSelectionCandidates(l_sortedNewSelectionBeam)


    def chooseSelectionCandidates(self,beam):
        newSelectionBeam = {}
        for rule in beam:
            newSelectionBeam[rule]=rule.selection_quality
        sorted_newSelectionBeam = sorted(newSelectionBeam.items(), key=operator.itemgetter(1), reverse=True)
        self.selectionCandidates = [i[0] for i in sorted_newSelectionBeam]

    def initializeRefinementBeam(self):
        self.RefinementBeam = [SDRule(data=self.data, targetClass=self.targetClass, g=self.g, refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)]*self.refinementBeamWidth

    def initializeSelectionBeam(self):
        self.SelectionBeam = [SDRule(data=self.data, targetClass=self.targetClass, g=self.g,refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)]*self.selectionBeamWidth

    def updateRefinementBeam(self, refinementCandidates):

        empty_rule = [SDRule(data=self.data, targetClass=self.targetClass, g=self.g, refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)]
        newRefinementBeam = {}
        alreadyRefined = []
        refinement_improvement = False

        for ref in self.RefinementBeam:
            if ref.complexity != 0:
                alreadyRefined.append(ref.orderedRuleToString())

        for refinement in refinementCandidates:
            if (refinement.orderedRuleToString() not in alreadyRefined):
                newRefinementBeam[refinement] = refinement.refinement_quality
                alreadyRefined.append(ref.orderedRuleToString())

        sorted_newRefinementBeam = sorted(newRefinementBeam.items(), key=operator.itemgetter(1), reverse=True)
        self.refinementCandidates = [i[0] for i in sorted_newRefinementBeam]

        if len(self.refinementCandidates) > self.refinementBeamWidth:
            self.RefinementBeam = self.refinementCandidates[:self.refinementBeamWidth]
        else:
            self.RefinementBeam = self.refinementCandidates + empty_rule*(self.refinementBeamWidth-len(self.refinementCandidates))

        return refinement_improvement

    def printBeam(self, beam, name):
        print "#"*100
        print "\n %s \n" %(name)
        for rule in beam:
            print "SQ: %.3f\tRQ: %.3f\tTP: %d\tFP: %d\t%s" %(rule.selection_quality,rule.refinement_quality,len(rule.TP),len(rule.FP),rule.ruleToString())
        print "*"*100

    def updateSelectionBeam(self, selectionCandidates):
        empty_rule = [SDRule(data=self.data, targetClass=self.targetClass, g=self.g, \
                             refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)]
        newSelectionBeam = {}
        changes = False
        alreadySelected = []

        for sel in self.SelectionBeam:
            if sel.complexity != 0:
                newSelectionBeam[sel] = sel.selection_quality
                alreadySelected.append(sel.orderedRuleToString())
                self.alreadySelectedRules.add(sel.orderedRuleToString())

        for selection in selectionCandidates:
            #if the selection quality is smaller than the worst rule for selection in SelectionBeam, ignore this rule and others that follow
            if selection.selection_quality < self.SelectionBeam[-1].selection_quality:
                break
            if (selection.orderedRuleToString() not in alreadySelected) and (selection.orderedRuleToString() not in self.alreadySelectedRules):
                newSelectionBeam[selection] = selection.selection_quality
                alreadySelected.append(selection.orderedRuleToString())
                changes = True

        sorted_newSelectionBeam = sorted(newSelectionBeam.items(), key=operator.itemgetter(1), reverse=True)
        self.selectionCandidates = [i[0] for i in sorted_newSelectionBeam]

        if len(self.selectionCandidates) > self.selectionBeamWidth:
            #the updated SelectionBeam should consist only of selectionBeamWidth elements
            self.sortSelectionCandidates(self.selectionCandidates)
            self.SelectionBeam = self.selectionCandidates[:self.selectionBeamWidth]
        else:
            self.SelectionBeam = self.selectionCandidates + empty_rule*(self.selectionBeamWidth-len(self.selectionCandidates))

        return changes

    def selectRelevantCandidates(self):
        empty_rule = [SDRule(data=self.data, targetClass=self.targetClass, g=self.g, \
                             refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)]
        i=1
        newSelectionBeam = [self.selectionCandidates[0]]
        for j in range(1,len(self.selectionCandidates)):
            candidate = self.selectionCandidates[j]
            if self.isRelevant(candidate, newSelectionBeam):
                i = i+1
                newSelectionBeam.append(candidate)

        if len(newSelectionBeam) > self.selectionBeamWidth:
            self.SelectionBeam = newSelectionBeam[:self.selectionBeamWidth]
        else:
            self.SelectionBeam = newSelectionBeam + empty_rule*(self.selectionBeamWidth-len(newSelectionBeam))

    def sortSelectionCandidates(self, selectionCandidates):
        sortedSelectionCandidates = []
        for i in range(len(selectionCandidates)-1):
            temps = {}
            for j in range(i+1,len(selectionCandidates)):
                if selectionCandidates[j]==selectionCandidates[j-1]:
                    temps[j-1] = selectionCandidates[j-1].TP
                    temps[j] = selectionCandidates[j-1].TP
                else:
                    temps[j-1] = selectionCandidates[j-1].TP
                    i = j
                    break
            sorted_temps = sorted(temps.items(), key=operator.itemgetter(1), reverse=True)
            l_sorted_temps = [k[0] for k in sorted_temps]
            for st in l_sorted_temps:
                sortedSelectionCandidates.append(selectionCandidates[st])

        if len(sortedSelectionCandidates)<len(selectionCandidates):
            sortedSelectionCandidates.append(selectionCandidates[-1])
        self.selectionCandidates = sortedSelectionCandidates

    def isRelevant(self, newRule, beam):
        for rule in beam:
            if newRule.isIrrelevant(rule):
                return false
        return true

    def refinedRefinementBeam(self, targetClass):
        min_sup = self.minSupport
        newRefinementCandidates = {}
        newSelectionCandidates = {}
        for refinement in self.RefinementBeam:
            if refinement.orderedRuleToString() not in self.alredyRefinedRules[targetClass] and refinement.complexity !=0:
                attributes = refinement.conditions()

                for attr in self.data.domain.attributes:
                    if attr.name not in attributes:
                        value = value = attr.firstvalue()
                        while value:
                            newRule = refinement.cloneAndAddCondition(attr,value,refinement,refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)
                            newRule.filterAndStore(refinement)
                            if newRule.support > min_sup:
                                #if (len(newRule.TP) != len(refinement.TP)) and (len(newRule.FP) != len(refinement.FP)):
                                if len(newRule.FP) < len(refinement.FP):
                                    newRefinementCandidates[newRule]=newRule.refinement_quality
                                    newSelectionCandidates[newRule]=newRule.selection_quality
                                    self.alredyRefinedRules[targetClass].add(refinement.orderedRuleToString())
                            value = attr.nextvalue(value)


        sorted_newRefinementCandidates = sorted(newRefinementCandidates.items(), key=operator.itemgetter(1), reverse=True)
        sorted_newSelectionCandidates = sorted(newSelectionCandidates.items(), key=operator.itemgetter(1), reverse=True)
        self.refinementCandidates = [i[0] for i in sorted_newRefinementCandidates]

        l_sortedSelectionCandidates = [i[0] for i in sorted_newSelectionCandidates]
        self.sortSelectionCandidates(l_sortedSelectionCandidates)

    def betterThanWorstRule(self, newRule, beam, worstRuleIndex):
        if newRule.quality2 > beam[worstRuleIndex].quality2:          # better quality
            return true
        elif newRule.quality2 == beam[worstRuleIndex].quality2 and newRule.complexity < beam[worstRuleIndex].complexity:   # same quality and smaller complexity
            return true
        else:
            return false

    def replaceWorstRule(self, rule, beam, worstRuleIndex):
        beam[worstRuleIndex] = rule
        wri = 0
        for i in range(len(beam)):
            if beam[i].quality2 < beam[wri].quality2:
                wri = i
        return wri

    def dataOK(self, data):
        if data.domain.classVar.varType != orange.VarTypes.Discrete:
            print "Target Variable must be discrete: %s"%(data.domain.classVar.name)
            return false
        return true

    def ruleSubsetSelection(self, beam, num_of_rules, data):
        SS = []
        c = orange.newmetaid()
        data.addMetaAttribute(c)   #initialize to 1
        if num_of_rules <= len(beam):
            for i in range(num_of_rules):
                best_score = 0
                best_rule_index = 0
                for i in range(len(beam)):
                    score = 0
                    for d in data:          # calculate sum of weights of examples
                        if beam[i].filter(d):
                            score += 1.0/d.getweight(c)
                    if score>best_score:
                        best_score = score
                        best_rule_index = i
                for d in data:              # increase exampe counter
                    if beam[best_rule_index].filter(d):
                        d.setweight(c, d.getweight(c)+1)
                SS.append(beam[best_rule_index])
                del beam[best_rule_index]
            data.removeMetaAttribute(c)

        else:
            return beam
        return SS

    def writeResults(self,file_name):
        current_directory = os.path.dirname(os.path.realpath(__file__)) + r"/results"
        print current_directory
#___________________________________________________________________________________
if __name__=="__main__":

    dataset_directory = current_directory = os.path.dirname(os.path.realpath(__file__))+r"/20_DATASETS_TAB/"
    dataset = "contact-lenses.tab"
    filename = os.path.join(dataset_directory,dataset)

    data = orange.ExampleTable(filename)
    print
    learner = DoubleBeam(minSupport=0.001, sb_width=1, refinement_heuristics = "Inverted Laplace", selection_heuristics="Laplace")
    rules = learner(data, targetClass="none", num_of_rules=0)
    learner.printBeam(learner.SelectionBeam, "SB")
    rules.printRules()


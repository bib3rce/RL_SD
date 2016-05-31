import Orange
import orange
import sys
import operator
from datetime import datetime
from SDRule_updated_weights import *

true = 1
false = 0

class DoubleBeam_weights:
    def __init__(self, type = "harmonic",weight_factor = 0.9, minSupport = 0.2, beam_width=10, g=1, refinement_heuristics = "Inverted Laplace", selection_heuristics = "Laplace", **kwds):
        self.minSupport = minSupport
        self.g = g
        self.CandidatesList = set()
        self.selectionBeamWidth = beam_width
        self.refinementBeamWidth = 500
        self.refinementCandidates = []
        self.selectionCandidates = []
        self.alredyRefinedRules = dict()
        self.refinement_heuristics = refinement_heuristics
        self.selection_heuristics = selection_heuristics
        self.alreadySelectedRules = set()
        #self.weights_type = "geometric"
        self.weights_type = type
        #self.weight_factor = 0.9
        self.weight_factor = weight_factor

    def __call__(self, data, targetClass, num_of_rules ):
        self.alredyRefinedRules[str(targetClass)] = set()
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
                        # An attribute is irrelevant, if it is discretized into a single interval
#                        if len(d_attribute.getValueFrom.transformer.points) > 0:
                        new_domain.append(d_attribute)
                    else:
                        new_domain.append(attribute)
                data = original_data.select(new_domain + [original_data.domain.classVar])

            self.data = data
            self.weigted_data = data
            self.c = orange.newmetaid()
            self.count = orange.newmetaid()
            self.weigted_data.addMetaAttribute(self.c)
            self.weigted_data.addMetaAttribute(self.count)
            #print self.c
            #print self.weigted_data.domain.attributes
            self.targetClass = targetClass

            #Initialize CanditatesList (all features)
            self.fillCandidatesList(data,targetClass)
            """
            print "Candidates for refinement:\n"
            for rule in self.refinementCandidates:
                print "N: %d\t\tTP: %d\t\t\tFP: %d\t\tRule:\t%s" %(len(rule.TP)+len(rule.FP),len(rule.TP), len(rule.FP), rule.ruleToString())

            print "\nCandidates for selection:\n"
            for rule in self.selectionCandidates:
                print "N: %d\t\tTP: %d\t\t\tFP: %d\t\tRule:\t%s" %(len(rule.TP)+len(rule.FP),len(rule.TP), len(rule.FP), rule.ruleToString())
            """
            """
            print self.refinementCandidates[0].ruleToString()
            print "Best refinement: P %d\tN %d\tp %d\tn %d\tRQ %.3f" %(self.refinementCandidates[0].P,self.refinementCandidates[0].N,len(self.refinementCandidates[0].TP),len(self.refinementCandidates[0].FP), self.refinementCandidates[0].refinement_quality)
            print "\n\n"
            """
            #Initialize RefinementBeam, consisting of refinementBeamWidth empty rules
            self.initializeRefinementBeam()
            #Initialize SelectionBeam, consisting of selectionBeamWidth empty rules
            self.initializeSelectionBeam()

            #update RefinementBeam
            self.updateRefinementBeam(self.refinementCandidates)
            #update SelectionBeam
            #self.chooseSelectionCandidates(self.RefinementBeam)
            """
            print self.selectionCandidates[0].ruleToString()
            print "Best selection: P %d\tN %d\tp %d\tn %d\tSQ %.3f" %(self.selectionCandidates[0].P,self.selectionCandidates[0].N,len(self.selectionCandidates[0].TP),len(self.selectionCandidates[0].FP), self.selectionCandidates[0].selection_quality)
            print "\n\n"
            """
            #print "Before updatation"
            self.updateSelectionBeam(self.selectionCandidates)
            #print "After update"

            #self.printBeam(self.refinementCandidates, name="Refinement candidates")
            #self.printBeam(self.RefinementBeam, name="Refinement beam")
            #self.printBeam(self.refinementCandidates, name="Refinement candidates")


            #self.printBeam(self.SelectionBeam, name="Selection beam")

            improvements = True
            refinement_improvements = True

            ms=2
            max_steps=5
            # and i<max_steps and refinement_improvements
            # improvements and i<max_steps and refinement_improvements:
            #while i<max_steps:
            while ms <= max_steps:
                #print "pocnuva rafiniranjeto, dolzina %d" %i
                self.refinedRefinementBeam(targetClass)
                #self.printBeam(self.refinementCandidates,"Refinement candidates")
                refinement_improvements = self.updateRefinementBeam(self.refinementCandidates)
                #self.printBeam(self.RefinementBeam, name="Refinement beam")
                #unionOfBeams = []; unionOfBeams.extend(self.RefinementBeam); unionOfBeams.extend(self.SelectionBeam)
                #self.chooseSelectionCandidates(unionOfBeams)
                #self.printBeam(self.selectionCandidates, name="Selection candidates")
                #print "Pred update"
                improvements = self.updateSelectionBeam(self.selectionCandidates)
                #print "Posle update"
                #m(self.SelectionBeam, "Selection beam")
                ms=ms+1

            beam = self.SelectionBeam
            #self.printBeam(beam, "Final selection beam.")
            if num_of_rules != 0:
                beam = self.ruleSubsetSelection(beam, num_of_rules, data)
                #self.printBeam(beam, "Posle SS")
                self.SelectionBeam = beam

            if data_discretized:
                 targetClassRule = SDRule(original_data, targetClass, conditions=[], g=self.g)
                 #targetClassRule = SDRule(original_data, targetClass, conditions=[], g=1, refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)
                 # change beam so the rules apply to original data
                 #self.printBeam(self.SelectionBeam, "Pred diskretizacija")
                 self.SelectionBeam = [rule.getUndiscretized(original_data) for rule in self.SelectionBeam]
                 #self.printBeam(self.SelectionBeam, "Posle diskretizacija")

            else:
                 targetClassRule = SDRule(data, targetClass, conditions=[], g =self.g)
                 #targetClassRule = SDRule(data, targetClass, conditions=[], g =1, refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)

            #print "Ready to return"
            #self.printBeam(self.SelectionBeam, "Ova se vrakja")
            rules = SDRules(self.SelectionBeam, targetClassRule, "SD-inverted")
            #rules.printRules()
            #print "*"*100
            return rules
            #return SDRules(self.SelectionBeam, targetClassRule, "SD-inverted")

    def fillCandidatesList(self, data, targetClass):
        #first initialize empty rule
        rule = SDRule(data=data, targetClass=targetClass, g=self.g, refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)
        newRefinementBeam = {}
        newSelectionBeam = {}

        self.alredyRefinedRules[str(targetClass)].add(rule.orderedRuleToString())

        for attr in data.domain.attributes:
            value = attr.firstvalue()
            while(value):
                #print i
                #i+=1
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
        #self.selectionCandidates = [i[0] for i in sorted_newSelectionBeam]


        #self.refinementBeamWidth = min(no_candidates*10,500)
        #self.refinementBeamWidth = 1

    def chooseSelectionCandidates(self,beam):
        newSelectionBeam = {}
        for rule in beam:
            newSelectionBeam[rule]=rule.selection_quality
        sorted_newSelectionBeam = sorted(newSelectionBeam.items(), key=operator.itemgetter(1), reverse=True)
        self.selectionCandidates = [i[0] for i in sorted_newSelectionBeam]

    def resetDataWeights(self):
        for d in self.weigted_data:
            d.setweight(self.c,1)
            d.setweight(self.count,0)

    def updateWeights(self,rule,type=""):
        #print rule
        #print rule.ruleToString()
        for d in self.weigted_data:
            if type=="geometric":
                if rule.covers(d):
                    weight = d.getweight(self.c)
                    weight = weight*self.weight_factor
                    d.setweight(self.c, weight)
                    #print "set weight: ", weight
                    #print "weight is set to: ", d.getweight(self.c)
                    #print "geometric: ", weight
                continue
            elif type=="harmonic":
                if rule.covers(d):
                    count = d.getweight(self.count)
                    count = count + 1
                    d.setweight(self.c, 1.0/count)
                    d.setweight(self.count, count)
                    #print "weight ", d.getweight(self.c)
                    #print "count ", d.getweight(self.count)
                    #print "set weight: ", weight
                    #print "weight is set to: ", d.getweight(self.c)
                    #print "harmonic: ", weight
                continue
            else:
                d.setweight(self.c, 0)
                #pass

    def replaceRefinementBeam(self, refinementCandidates):
        empty_rule = [SDRule(data=self.data, targetClass=self.targetClass, g=self.g, refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)]
        newRefinementBeam = {}
        alreadyRefined = []
        refinement_improvement = False

        #for ref in self.RefinementBeam:
        #    if ref.complexity != 0:
                #newRefinementBeam[ref] = ref.refinement_quality
        #        alreadyRefined.append(ref.orderedRuleToString())

        for refinement in refinementCandidates:
            #if the refinemet quality is smaller than the worst rule for refinement in RefinementBeam, ignore this rule and others that follow
            #if refinement.refinement_quality < self.RefinementBeam[-1].refinement_quality:
            #    print "we are breaking"
            #    break
            if (refinement.orderedRuleToString() not in alreadyRefined):
                #otherwise insert the refinement into the right position in newRefinementBeam
                newRefinementBeam[refinement] = refinement.refinement_quality
                alreadyRefined.append(refinement.orderedRuleToString())

        sorted_newRefinementBeam = sorted(newRefinementBeam.items(), key=operator.itemgetter(1), reverse=True)
        self.refinementCandidates = [i[0] for i in sorted_newRefinementBeam]

        if len(self.refinementCandidates) > self.refinementBeamWidth:
            #the updated RefinementBeam should consist only of refinementBeamWidth elements
            self.RefinementBeam = self.refinementCandidates[:self.refinementBeamWidth]
        else:
            self.RefinementBeam = self.refinementCandidates + empty_rule*(self.refinementBeamWidth-len(self.refinementCandidates))

        for i in range(min(len(self.refinementCandidates), self.refinementBeamWidth)):
            if self.refinementCandidates[i].orderedRuleToString() not in alreadyRefined:
                refinement_improvement = True

        return refinement_improvement

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
                #newRefinementBeam[ref] = ref.refinement_quality
                alreadyRefined.append(ref.orderedRuleToString())

        for refinement in refinementCandidates:
            #if the refinemet quality is smaller than the worst rule for refinement in RefinementBeam, ignore this rule and others that follow
            #if refinement.refinement_quality < self.RefinementBeam[-1].refinement_quality:
            #    print "we are breaking"
            #    break
            if (refinement.orderedRuleToString() not in alreadyRefined):
                #otherwise insert the refinement into the right position in newRefinementBeam
                newRefinementBeam[refinement] = refinement.refinement_quality
                alreadyRefined.append(ref.orderedRuleToString())

        sorted_newRefinementBeam = sorted(newRefinementBeam.items(), key=operator.itemgetter(1), reverse=True)
        self.refinementCandidates = [i[0] for i in sorted_newRefinementBeam]

        if len(self.refinementCandidates) > self.refinementBeamWidth:
            #the updated RefinementBeam should consist only of refinementBeamWidth elements
            self.RefinementBeam = self.refinementCandidates[:self.refinementBeamWidth]
        else:
            self.RefinementBeam = self.refinementCandidates + empty_rule*(self.refinementBeamWidth-len(self.refinementCandidates))

        """
        for i in range(min(len(self.refinementCandidates), self.refinementBeamWidth)):
            if self.refinementCandidates[i].orderedRuleToString() not in alreadyRefined:
                refinement_improvement = True

        """
        return refinement_improvement
        """
        print "*-"*50
        print "\n"
        print "Rule to be refined: ", self.RefinementBeam[0].ruleToString()
        print "\n"
        print "*-"*50
        """

    def printBeam(self, beam, name):
        print "#"*100
        print "\n %s \n" %(name)
        for rule in beam:
            print "SQ: %.3f\tRQ: %.3f\tTP: %d\tFP: %d\t%s" %(rule.selection_quality,rule.refinement_quality,len(rule.TP),len(rule.FP),rule.ruleToString())
        print "*"*100

    def updateSelectionBeam(self, selectionCandidates):
        empty_rule = [SDRule(data=self.data, targetClass=self.targetClass, g=self.g, \
                             refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)]
        newSelectionBeam = list()
        #newSelectionBeam = {}
        changes = False
        alreadySelected = list()

        self.resetDataWeights()

        candidates = list()
        for c in self.SelectionBeam:
            if c.complexity != 0:
                if c.orderedRuleToString() not in alreadySelected:
                    candidates.append(c)
                    alreadySelected.append(c.orderedRuleToString())


        for c in selectionCandidates:
            if c.complexity != 0:
                if c.orderedRuleToString() not in alreadySelected:
                    candidates.append(c)
                    alreadySelected.append(c.orderedRuleToString())

        for i in range(self.selectionBeamWidth):
            if candidates==[]:
                break
            bestRule = self.selectBestRule(candidates)
            #print "best rule: ", bestRule.ruleToString()
            #print "TP: ", bestRule.TPlen
            #print "FP: ", bestRule.FPlen
            #print "pred updateni"
            self.updateWeights(rule=bestRule, type=self.weights_type)
            #print "posle updatani"
            #print "Weights are updated"
            candidates.remove(bestRule)
            newSelectionBeam.append(bestRule)


        if len(newSelectionBeam) < self.selectionBeamWidth:
            self.SelectionBeam = newSelectionBeam + empty_rule*(self.selectionBeamWidth-len(newSelectionBeam))
        else:
            self.SelectionBeam = newSelectionBeam

        """
        #print "New selection beam."
        for sel in self.SelectionBeam:
            if sel.complexity != 0:
                newSelectionBeam[sel] = sel.selection_quality
                alreadySelected.append(sel.orderedRuleToString())
                self.alreadySelectedRules.add(sel.orderedRuleToString())

        for selection in selectionCandidates:
            #print selection.orderedRuleToString()
            #if the selection quality is smaller than the worst rule for selection in SelectionBeam, ignore this rule and others that follow
            if selection.selection_quality < self.SelectionBeam[-1].selection_quality:
                break
            if (selection.orderedRuleToString() not in alreadySelected) and (selection.orderedRuleToString() not in self.alreadySelectedRules):
                #otherwise insert the refinement into the right position in newRefinementBeam
                newSelectionBeam[selection] = selection.selection_quality
                alreadySelected.append(selection.orderedRuleToString())
                #self.alreadySelectedRules.add(selection.orderedRuleToString())
                changes = True
            #else:
            #    print selection.orderedRuleToString(), ' is in SelectionBeam.'

        #print "*"*39

        sorted_newSelectionBeam = sorted(newSelectionBeam.items(), key=operator.itemgetter(1), reverse=True)
        self.selectionCandidates = [i[0] for i in sorted_newSelectionBeam]

        if len(self.selectionCandidates) > self.selectionBeamWidth:
            #the updated SelectionBeam should consist only of selectionBeamWidth elements
            self.sortSelectionCandidates(self.selectionCandidates)
            #self.selectRelevantCandidates()
            self.SelectionBeam = self.selectionCandidates[:self.selectionBeamWidth]
            #self.SelectionBeam = []

            #for sel in self.selectionCandidates:
            #    if

        else:
            self.SelectionBeam = self.selectionCandidates + empty_rule*(self.selectionBeamWidth-len(self.selectionCandidates))

        #print len(self.SelectionBeam)
        return changes


        empty_rule = [SDRule(data=self.data, targetClass=self.targetClass, g=self.g, refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)]
        newSelectionBeam = {}
        changes = False
        alreadySelected = []
        alreadyAdded = []

        #print "New selection beam."
        for sel in self.SelectionBeam:
            if sel.complexity != 0:
                #newSelectionBeam[sel] = sel.selection_quality
                alreadySelected.append(sel.orderedRuleToString())
                alreadyAdded.append(sel)

        for selection in selectionCandidates:
            #print selection.orderedRuleToString()
            #if the selection quality is smaller than the worst rule for selection in SelectionBeam, ignore this rule and others that follow
            if selection.selection_quality < self.SelectionBeam[-1].selection_quality:
                break
            if (selection.orderedRuleToString() not in alreadySelected):
                #otherwise insert the refinement into the right position in newRefinementBeam
                if len(alreadyAdded)==0:
                    #newSelectionBeam[selection] = selection.selection_quality
                    alreadySelected.append(selection.orderedRuleToString())
                    alreadyAdded.append(selection)
                    changes = True
                else:
                    if selection.selection_quality == alreadyAdded[-1].selection_quality:
                        #print "two rules with same quality"
                        if len(selection.filter.conditions) < len(alreadyAdded[-1].filter.conditions):
                            #print selection.conditions
                            #print "we change now"
                            #self.printBeam(alreadyAdded,name="already added before removing")
                            alreadyAdded.remove(alreadyAdded[-1])
                            #self.printBeam(alreadyAdded,name="already added after removing")
                            alreadySelected.append(selection.orderedRuleToString())
                            alreadyAdded.append(selection)
                            #self.printBeam(alreadyAdded,name="already added after adding")
                            changes = True
                    else:
                        alreadySelected.append(selection.orderedRuleToString())
                        alreadyAdded.append(selection)
                        changes = True

        for s in alreadyAdded:
            newSelectionBeam[s] = s.selection_quality

            #else:
            #    print selection.orderedRuleToString(), ' is in SelectionBeam.'

        #print "*"*39

        sorted_newSelectionBeam = sorted(newSelectionBeam.items(), key=operator.itemgetter(1), reverse=True)
        self.selectionCandidates = [i[0] for i in sorted_newSelectionBeam]

        if len(self.selectionCandidates) > self.selectionBeamWidth:
            #the updated SelectionBeam should consist only of selectionBeamWidth elements
            self.SelectionBeam = self.selectionCandidates[:self.selectionBeamWidth]
        else:
            self.SelectionBeam = self.selectionCandidates + empty_rule*(self.selectionBeamWidth-len(self.selectionCandidates))

        #print len(self.SelectionBeam)
        return changes
        """
        """
        tempSelectionBeam = []
        for i in range(len(self.selectionCandidates)-1):
            temp_i = i
            if sorted_newSelectionBeam[self.selectionCandidates[i]]==sorted_newSelectionBeam[self.selectionCandidates[i+1]]:
                temp = {}
                temp[self.selectionCandidates[i]]=len(self.selectionCandidates[i].conditions)
                for j in range(i+1,len(self.selectionCandidates)-1):
                    if sorted_newSelectionBeam[self.selectionCandidates[i]]==sorted_newSelectionBeam[self.selectionCandidates[j]]:
                        temp[self.selectionCandidates[j]]=len(self.selectionCandidates[i].conditions)
                    else:
                        temp_i = j
                        break

                sorted_temp = sorted(temp.items(), key=operator.itemgetter(1), reverse=False)
                tempSelectionBeam.append(sorted_temp[0])
            else:
                tempSelectionBeam.append(self.selectionCandidates[i])

        if len(tempSelectionBeam) > self.selectionBeamWidth:
            #the updated SelectionBeam should consist only of selectionBeamWidth elements
            self.SelectionBeam = tempSelectionBeam[:self.selectionBeamWidth]
        else:
            self.SelectionBeam = tempSelectionBeam + empty_rule*(self.selectionBeamWidth-len(self.selectionCandidates))

        print "*-"*50
        print "\n"
        print "Rule to be selected: ", self.SelectionBeam[0].ruleToString()
        print "\n"
        print "*-"*50
        """
        #return changes

    def selectBestRule(self,candidates):
        bestRule = []
        alreadyChecked = list()

        if len(candidates)>0:
            bestRule = candidates[0]
            bestRule.setWeightedData(self.weigted_data); bestRule.calculateWeightedTP(self.c); #bestRule.calculateWeightedFP(self.c);
            bestRule.calculateWeightedSelectionQuality()
            bestQuality=bestRule.weighted_selection_quality
            alreadyChecked.append(bestRule.orderedRuleToString())


            for i in range(1,len(candidates)):
                rule = candidates[i]
                rule.setWeightedData(self.weigted_data); rule.calculateWeightedTP(self.c); #rule.calculateWeightedFP(self.c);
                rule.calculateWeightedSelectionQuality()
                ruleQuality=rule.weighted_selection_quality
                if rule.orderedRuleToString() not in alreadyChecked:
                    if ruleQuality > bestQuality:
                        bestQuality = ruleQuality
                        bestRule = rule
                    elif ruleQuality == bestQuality:
                        if rule.TPlen > bestRule.TPlen:
                            bestQuality = ruleQuality
                            bestRule = rule
                        elif rule.TPlen == bestRule.TPlen:
                            if rule.complexity < bestRule.complexity:
                                bestQuality = ruleQuality
                                bestRule = rule

                    alreadyChecked.append(rule.orderedRuleToString())

        return bestRule

    def selectRelevantCandidates(self):
        empty_rule = [SDRule(data=self.data, targetClass=self.targetClass, g=self.g, \
                             refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)]
        i=1
        newSelectionBeam = [self.selectionCandidates[0]]
        for j in range(1,len(self.selectionCandidates)):
            candidate = self.selectionCandidates[j]
            if self.isRelevant(candidate, newSelectionBeam):
                #print "Relevant rule ", candidate.ruleToString()
                i = i+1
                newSelectionBeam.append(candidate)
            #else:
                #print "Irrelevant rule ", candidate.ruleToString()

            #if i == self.selectionBeamWidth:
            #    break

        if len(newSelectionBeam) > self.selectionBeamWidth:
            self.SelectionBeam = newSelectionBeam[:self.selectionBeamWidth]
        else:
            self.SelectionBeam = newSelectionBeam + empty_rule*(self.selectionBeamWidth-len(newSelectionBeam))

        #self.printBeam(self.SelectionBeam, "Updated selection beam")

    def sortSelectionCandidates(self, selectionCandidates):
        #print "selection candidates", len(selectionCandidates)
        sortedSelectionCandidates = []
        for i in range(len(selectionCandidates)-1):
            #candidate = selectionCandidates[i]
            #ref_candidate = selectionCandidates[i+1]
            #if candidate.selection_quality == ref_candidate.selection_quality:
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
        """
            else:
                sortedSelectionCandidates.append(selectionCandidates[i])

        if selectionCandidates
        """

        #print "Sorted selection candidates ", len(sortedSelectionCandidates)
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
            if refinement.orderedRuleToString() not in self.alredyRefinedRules[str(targetClass)] and refinement.complexity !=0:
                attributes = refinement.conditions()

                for attr in self.data.domain.attributes:
                    if attr.name not in attributes:
                        value = value = attr.firstvalue()
                        while value:
                            newRule = refinement.cloneAndAddCondition(attr,value,refinement,refinement_heuristics=self.refinement_heuristics, selection_heuristics=self.selection_heuristics)
                            newRule.filterAndStore(refinement)
                            #print newRule.ruleToString()
                            #print newRule.support
                            if newRule.support > min_sup:
                                #if (len(newRule.TP) != len(refinement.TP)) and (len(newRule.FP) != len(refinement.FP)):
                                if len(newRule.FP) < len(refinement.FP):
                                    newRefinementCandidates[newRule]=newRule.refinement_quality
                                    newSelectionCandidates[newRule]=newRule.selection_quality
                                    self.alredyRefinedRules[str(targetClass)].add(refinement.orderedRuleToString())
                            value = attr.nextvalue(value)


        sorted_newRefinementCandidates = sorted(newRefinementCandidates.items(), key=operator.itemgetter(1), reverse=True)
        sorted_newSelectionCandidates = sorted(newSelectionCandidates.items(), key=operator.itemgetter(1), reverse=True)
        self.refinementCandidates = [i[0] for i in sorted_newRefinementCandidates]

        l_sortedSelectionCandidates = [i[0] for i in sorted_newSelectionCandidates]
        self.sortSelectionCandidates(l_sortedSelectionCandidates)
        #self.selectionCandidates = [i[0] for i in sorted_newSelectionCandidates]
        """
        if len(self.refinementCandidates)!=0:
            print "#"*100
            print self.refinementCandidates[0].ruleToString()
            print "Best refinement: P %d\tN %d\tp %d\tn %d\tRQ %.3f" %(self.refinementCandidates[0].P,self.refinementCandidates[0].N,len(self.refinementCandidates[0].TP),len(self.refinementCandidates[0].FP), self.refinementCandidates[0].refinement_quality)
            print "\n\n"
        """
        """
            print self.selectionCandidates[0].ruleToString()
            print "Best selection: P %d\tN %d\tp %d\tn %d\tSQ %.3f" %(self.selectionCandidates[0].P,self.selectionCandidates[0].N,len(self.selectionCandidates[0].TP),len(self.selectionCandidates[0].FP), self.selectionCandidates[0].selection_quality)
            print "\n\n"
        """

        """
        print "Candidates for refinement: \n"
        for rule in self.refinementCandidates:
                print "N: %d\t\tTP: %d\t\t\tFP: %d\t\tRule:\t%s" %(len(rule.TP)+len(rule.FP),len(rule.TP), len(rule.FP), rule.ruleToString())

        print "#"*80

        print "\n\n"

        print "Candidates for selection: \n"
        for rule in self.selectionCandidates:
                print "N: %d\t\tTP: %d\t\t\tFP: %d\t\tRule:\t%s" %(len(rule.TP)+len(rule.FP),len(rule.TP), len(rule.FP), rule.ruleToString())

        print "#"*80
        """

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
            print "Target Variable must be discrete"%(attr.name)
            return false
        return true

    def ruleSubsetSelection(self, beam, num_of_rules, data):
        #print "RSS"
        #self.printBeam(beam, "RSS")
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
    learner = DoubleBeam_weights(weight_factor = 0.9,minSupport=0.01, beam_width=5, refinement_heuristics = "Inverted precision", selection_heuristics="Precision", type="harmonic")
    """
    for targetClass in data.domain.classVar.values:
        #targetClass= orange.Value(data.domain.classVar, "hard")
        print targetClass
        rules = learner(data, targetClass=targetClass, num_of_rules=5)
        rules.printRules()
    """
    rules = learner(data, targetClass="hard", num_of_rules=0)
    learner.printBeam(learner.SelectionBeam, "SB")
    rules.printRules()
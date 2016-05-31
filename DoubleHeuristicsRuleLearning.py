__author__ = 'anitavalmarska'

from DoubleBeam_Ref import *
from SDRule import *
import Orange

class RuleLearning:
    def __init__(self, data = None, refinement_heuristics = "Inverted Laplace", selection_heuristics = 'Laplace', rb_width=1, sb_width=1, g=1):
        if data == None:
            print "Please, provide a dataset."
            pass

        self.originalData = data
        self.sb_width = sb_width
        self.rb_width = rb_width
        self.alreadyCoveredExamples = dict()
        self.examples = list()
        self.examplesByClass = dict()
        self.refinementHeuristics = refinement_heuristics
        self.selectionHeuristics = selection_heuristics
        self.ruleSets = dict()
        self.build_rule_sets()
        #self.print_rule_sets()

    def build_rule_sets(self):
        self.get_examples_ids()
        for targetClass in self.originalData.domain.classVar.values:
            self.alreadyCoveredExamples[targetClass] = []
            ruleSet = self.learn_rule_set(targetClass=targetClass)
            self.ruleSets[targetClass] = ruleSet

    def learn_rule_set(self, targetClass = None):
        ruleSet = []
        if targetClass == None:
            print "In order to be able to learn something, we need a value for the class variable."
            return ruleSet

        satisfied = False
        tempData = self.originalData
        ref_num = len(tempData)

        while not satisfied:
            learner = DoubleBeam(minSupport=0.0001, sb_width=self.sb_width, rb_width=self.rb_width,\
                                 refinement_heuristics = self.refinementHeuristics, selection_heuristics=self.selectionHeuristics)
            rules = learner(tempData, targetClass=targetClass, num_of_rules=0)
            bestRule = self.select_best_rule(rules)
            tempData = self.remove_positive_examples(data=tempData, rule=bestRule, targetClass=targetClass)
            new_ref = len(tempData)

            if ref_num == new_ref:
                satisfied = True
            else:
                if self.acceptable_rule(rule=bestRule):
                    ruleSet.append(bestRule)
                    ref_num = new_ref
                else:
                    satisfied=True
        return ruleSet

    def select_best_rule(self, sd_rules):
        bestRule = sd_rules.rules[0]

        return bestRule

    def acceptable_rule(self, rule=None):
        if rule==None:
            return False

        if rule.FPlen > rule.TPlen:
            return False
        elif rule.TPlen <= 2:
            return False
        else:
            return True

    def remove_positive_examples(self, data=None, rule=None, targetClass=None):
        if data == None:
            print "Data cannot be of type None. Please provide a dataset."
            return []
        if rule == None:
            print "Please provide a rule you learnt."
            return []
        if targetClass == None:
            print "Please provide a rule you learnt."
            return []

        covered_TP = self.get_covered_positive_examples(data,rule,targetClass)
        alreadyCovered = self.alreadyCoveredExamples[targetClass]
        totalCoveredExamples = alreadyCovered+covered_TP
        self.alreadyCoveredExamples[targetClass]=totalCoveredExamples
        uncoveredExamples = list(set(self.examples)-set(totalCoveredExamples))
        updated_data = Orange.data.utils.take(self.originalData, uncoveredExamples, axis=0)
        return updated_data

    def get_covered_positive_examples(self, data=None, rule=None, targetClass=None):
        if data == None:
            print "Data cannot be of type None. Please provide a dataset."
            return []
        if rule == None:
            print "Please provide a rule you learnt."
            return []
        if targetClass == None:
            print "Please provide a rule you learnt."
            return []

        TP_indexes = []
        for i in range(len(self.originalData)):
            example = self.originalData[i]
            if rule.filter(example) and example.getclass()==targetClass:
                TP_indexes.append(i)
        return TP_indexes

    def check_satisfied(self, data=None, targetClass=None):
        if data == None:
            print "Data cannot be of type None. Please provide a dataset."
            return True
        if targetClass == None:
            print "Please provide a rule you learnt."
            return True

        positives = filter(lambda e: e.getclass()==targetClass, data)
        if len(positives) == 0:
            return True
        else:
            return False

    def get_examples_by_class(self, targetClass):
        TP_indexes = []
        for i in range(len(self.originalData)):
            e = self.originalData[i]
            if e.getclass()==targetClass:
                TP_indexes.append(i)

        self.examplesByClass[targetClass] = TP_indexes

    def get_examples_ids(self):
        for i in range(len(self.originalData)):
            self.examples.append(i)

    def print_rule_sets(self):
        for targetClass in self.ruleSets:
            ruleSet = self.ruleSets[targetClass]
            print "#"*100
            print "These are the rules for: ", targetClass
            for rule in ruleSet:
                print "SQ: %.3f\tRQ: %.3f\tTP: %d\tFP: %d\t%s" %(rule.selection_quality,rule.refinement_quality,len(rule.TP),len(rule.FP),rule.ruleToString())

            print "*"*100
            print "\n"




if __name__ == "__main__":
    dataset_directory = current_directory = os.path.dirname(os.path.realpath(__file__))+r"/20_DATASETS_TAB/"
    dataset = "contact-lenses.tab"
    filename = os.path.join(dataset_directory,dataset)

    data = orange.ExampleTable(filename)
    rl = RuleLearning(data=data, selection_heuristics="Laplace", refinement_heuristics="Inverted Laplace", sb_width=1, rb_width=10)









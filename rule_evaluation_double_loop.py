import Orange
import orange
from DoubleHeuristicsRuleLearning import *
import datetime
import os
from joblib import Parallel, delayed
import multiprocessing
from numpy import array

base_address = os.path.dirname(os.path.realpath(__file__)) +r"/datasets_separated/"
writing_base_address = os.path.dirname(os.path.realpath(__file__))+r"/DB_RL/double_loop/testing_results/partial_results/"


sheet_names = {"Laplace & Laplace":"LL",
                            "Laplace & Inverted Laplace":"LIL",
                            "Laplace & Precision":"LP",
                            "Laplace & Inverted precision":"LIP",
                            "Laplace & m-est":"LM",
                            "Laplace & Inverted m":"LIM",
                            "Laplace & WRACC":"LW",
                            "Inverted Laplace & Laplace":"ILL",
                            "Inverted Laplace & Inverted Laplace":"ILIL",
                            "Inverted Laplace & Precision":"ILP",
                            "Inverted Laplace & Inverted precision":"ILIP",
                            "Inverted Laplace & m-est":"ILM",
                            "Inverted Laplace & Inverted m":"ILIM",
                            "Inverted Laplace & WRACC":"ILW",
                            "Precision & Laplace":"PL",
                            "Precision & Inverted Laplace":"PIL",
                            "Precision & Precision":"PP",
                            "Precision & Inverted precision":"PIP",
                            "Precision & m-est":"PM",
                            "Precision & Inverted m":"PIM",
                            "Precision & WRACC":"PW",
                            "Inverted precision & Laplace":"IPL",
                            "Inverted precision & Inverted Laplace":"IPIL",
                            "Inverted precision & Precision":"IPP",
                            "Inverted precision & Inverted precision":"IPIP",
                            "Inverted precision & m-est":"IPM",
                            "Inverted precision & Inverted m":"IPIM",
                            "Inverted precision & WRACC":"IPW",
                            "m-est & Laplace":"ML",
                            "m-est & Inverted Laplace":"MIL",
                            "m-est & Precision":"MP",
                            "m-est & Inverted precision":"MIP",
                            "m-est & m-est":"MM",
                            "m-est & Inverted m":"MIM",
                            "m-est & WRACC":"MW",
                            "Inverted m & Laplace":"IML",
                            "Inverted m & Inverted Laplace":"IMIL",
                            "Inverted m & Precision":"IMP",
                            "Inverted m & Inverted precision":"IMIP",
                            "Inverted m & m-est":"IMM",
                            "Inverted m & Inverted m":"IMIM",
                            "Inverted m & WRACC":"IMW",
                            "WRACC & Laplace":"WL",
                            "WRACC & Inverted Laplace":"WIL",
                            "WRACC & Precision":"WP",
                            "WRACC & Inverted precision":"WIP",
                            "WRACC & m-est":"WM",
                            "WRACC & Inverted m":"WIM",
                            "WRACC & WRACC":"WW"}

class RuleEvaluation:
    def __init__(self, dataset, fold, sh, rh, sb, rb):
        self.start_time = datetime.datetime.now()
        rsh_comb = rh+" & "+sh
        ref_sel_beam = "ref_"+str(rb)+"_sel_"+str(sb)
        self.address = base_address+dataset
        self.writing_address = writing_base_address+dataset+"/"+sheet_names[rsh_comb]
        self.dataset = dataset
        self.sh = sh
        self.rh = rh
        self.sb = int(sb)
        self.rb = int(rb)
        self.fold = fold
        self.rb_candidates = [1,5,10,20,50,75,100,500]

        self.read_learn_data()
        self.read_test_data()

        self.sort_classes()
        self.set_majority_class()

        #self.get_best_parameter(self):
        self.learn_rules()
        self.score_model()

        self.end_time = datetime.datetime.now()

        self.write_results()


    def read_learn_data(self):
        dataset_address = self.address+"/learn"
        file_name = str(self.fold)+".tab"

        file_address = os.path.join(dataset_address,file_name)
        self.learndata = orange.ExampleTable(file_address)

        self.trainingData, self.evaluationSet = self.splitTrainData(self.learndata)

    def read_test_data(self):
        dataset_address = self.address+"/test"
        file_name = str(self.fold)+".tab"

        file_address = os.path.join(dataset_address,file_name)
        self.testdata = orange.ExampleTable(file_address)

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

    def learn_rules(self):

        rl = RuleLearning(data=self.learndata, refinement_heuristics=self.rh, \
                        selection_heuristics=self.sh, rb_width=self.rb, sb_width=self.sb)

        decisionList = rl.ruleSets

        self.model = decisionList

    def score_model(self):
        TP = 0; FP = 0

        N = len(self.testdata)
        uncovered_examples = []

        for d in self.testdata:
            covered = False

            predicted_class, covered = self.predict_example_class(d)
            if not covered:
                uncovered_examples.append(d)

            if d.getclass() == predicted_class:
                TP = TP+1
            else:
                FP = FP+1


        UE = len(uncovered_examples)

        self.TP = TP
        self.FP = FP

        self.CA = self.calculate_CA(TP=TP,N=N)
        self.coverage = self.calculate_coverage(UE = UE, N=N)

        self.get_total_number_of_rules()
        self.get_average_rule_length()

    def get_best_parameter(self):
        max_CA = 0.0
        rb = 1
        for rbc in self.rb_candidates:
            rl = RuleLearning(data=self.trainingData, refinement_heuristics=self.rh, \
                              selection_heuristics=self.sh, rb_width=rbc, sb_width=self.sb)

            decisionList = rl.ruleSets

            CA = self.score_model_p(model=decisionList, evaluationData=self.evaluationSet)
            if CA > max_CA:
                max_CA = CA
                rb = rbc

        self.rb = rb

    def score_model_p(self, model, evaluationData):
        TP = 0;
        FP = 0

        N = len(evaluationData)

        for d in evaluationData:
            covered = False

            predicted_class, covered = self.predict_example_class_p(d,model)

            if d.getclass() == predicted_class:
                TP = TP + 1
            else:
                FP = FP + 1

        CA = self.calculate_CA(TP=TP, N=N)
        return CA


    def predict_example_class_p(self, example, model):
        covered = False

        distribution = dict()
        for cls in model:
            distribution[cls] = 0

        for cls in self.sorted_class_variables_p:
            rules = model[cls]
            for rule in rules:
                if rule.covers(example):
                    return cls, True

        return self.majority_class_p, False

    def predict_example_class(self, example):
        covered = False

        distribution = dict()
        for cls in self.model:
            distribution[cls] = 0

        for cls in self.sorted_class_variables:
            rules = self.model[cls]
            for rule in rules:
                if rule.covers(example):
                    return cls, True

        return self.majority_class, False

    def sort_classes(self):
        distribution = Orange.statistics.distribution.Distribution(self.learndata.domain.class_var.name, self.learndata)
        class_vars = self.learndata.domain.class_var.values

        dict_distr = {i:distribution[i] for i in range(len(distribution))}
        sorted_distribution = sorted(dict_distr.items(), key=operator.itemgetter(1), reverse=False)
        #print sorted_distribution

        indexes = [i[0] for i in sorted_distribution]

        sorted_class_vars = [class_vars[i] for i in indexes]

        self.sorted_class_variables = sorted_class_vars

    def sort_classes_p(self):
        distribution = Orange.statistics.distribution.Distribution(self.trainingData.domain.class_var.name, self.trainingData)
        class_vars = self.trainingData.domain.class_var.values

        dict_distr = {i: distribution[i] for i in range(len(distribution))}
        sorted_distribution = sorted(dict_distr.items(), key=operator.itemgetter(1), reverse=False)

        #print sorted_distribution

        indexes = [i[0] for i in sorted_distribution]

        sorted_class_vars = [class_vars[i] for i in indexes]

        self.sorted_class_variables_p = sorted_class_vars

    def set_majority_class(self):
        distribution = Orange.statistics.distribution.Distribution(self.learndata.domain.class_var.name, self.learndata)
        class_vars = self.learndata.domain.class_var.values

        max_distr_index = list(distribution).index(max(distribution))
        self.majority_class = class_vars[max_distr_index]

        distribution = Orange.statistics.distribution.Distribution(self.trainingData.domain.class_var.name, self.trainingData)
        class_vars = self.trainingData.domain.class_var.values

        max_distr_index = list(distribution).index(max(distribution))
        self.majority_class_p = class_vars[max_distr_index]

    def calculate_coverage(self, UE=0, N=1):
        coverage = 1.0*(N-UE)/N
        return coverage

    def calculate_CA(self, TP=0, N=1):
        CA = float(1.0*TP/N)
        return CA

    def get_total_number_of_rules(self):
        nr = 0

        for targetClass in self.model:
            nr = nr + len(self.model[targetClass])

        self.NR = nr

    def get_average_rule_length(self):
        nr = 0
        length = 0

        for targetClass in self.model:
            rules = self.model[targetClass]
            nr = nr + len(rules)
            for rule in rules:
                length = length + rule.complexity

        if nr !=0:
            average_rule_length = 1.0*length/nr
        else:
            average_rule_length = 0.0

        self.ARL = average_rule_length

    def write_results(self):

        if not os.path.exists(self.writing_address):
            os.makedirs(self.writing_address)

        filename = os.path.join(self.writing_address, self.fold+".txt")
        f = open(filename, "w")

        line = "MEASURE\tVALUE\n\n"
        f.write(line)

        line = "CA\t"+"{0:.3f}".format(self.CA)+"\n"
        f.write(line)

        line = "Coverage\t"+"{0:.3f}".format(self.coverage)+"\n"
        f.write(line)

        line = "NR\t"+str(self.NR)+"\n"
        f.write(line)

        line = "ARL\t"+"{0:.3f}".format(self.ARL)+"\n"
        f.write(line)

        line = "TP\t"+str(self.TP)+"\n"
        f.write(line)

        line = "FP\t"+str(self.FP)+"\n"
        f.write(line)

        line = "RB\t" + str(self.rb) + "\n"
        f.write(line)

        diff = self.end_time - self.start_time
        run_time_mins = divmod(diff.days*86400+diff.seconds,60)
        f.write("RT\t"+str(diff)+"\n")

        f.close()


def call_rule_evaluation(dataset, fold, sh, rh, sb, rb):
    RuleEvaluation(dataset, fold, sh, rh, sb, rb)

num_cores = multiprocessing.cpu_count()


combinations = []
learners = [['m-est','m-est'],['Inverted Laplace','m-est'],['WRACC','m-est']]


for dataset in ['contact-lenses']:
    for learner in learners:
        for fold in range(10):
            combinations.append([dataset, str(fold),learner[1],learner[0],'1','1'])

Parallel(n_jobs=num_cores)(delayed(call_rule_evaluation)(combinations[k][0], combinations[k][1], combinations[k][2],combinations[k][3], combinations[k][4], combinations[k][5]) for k in range(len(combinations)))




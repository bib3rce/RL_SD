import orange
import sys

def Learner( examples=None, **kwds):
    learner = apply(Learner_Class, (), kwds)
    if examples:
        return learner(examples, targetClass)
    else:
        return learner


class Learner_Class:
    def __init__(self, targetClass ,name='TargetClassClassifier'):
        self.name = name
        self.targetClass = targetClass
        #print "Learner class"

    def __call__(self, data):
        '''Returns a classifier tha always classifiers in the target class.'''
        #print self.targetClass
        distribution = [0.0] * len(data.domain.classVar.values)
        #print "Data: ", len(data)
        #print distribution
        #i =0
        for d in data:
            distribution[int(d.getclass())]+=1
        l=len(data)-0.000001
        #print distribution
        distribution = map(lambda x: x /l, distribution)
        #print "Distribution: ", distribution
        return Classifier( self.targetClass , distribution)

class Classifier:
    def __init__(self, value, distribution):
        #print "Classifier"
        self.defaultVal = value
        self.distribution = distribution

    def __call__(self, example, resultType = orange.GetValue):
        if resultType == orange.GetValue :
            return self.defaultVal
        elif resultType == orange.GetBoth :            
            return (self.defaultVal, self.distribution)
        else :
            return self.distribution

if __name__=="__main__":
    filename = "..\\..\\doc\\datasets\\lenses.tab"
    if 'linux' in sys.platform:
        filename= "/usr/doc/orange/datasets/lenses.tab"
    data = orange.ExampleTable(filename)

    learner = Learner(targetClass= orange.Value(data.domain.classVar, "hard"), name = "TargetClassClassifier")
    classifier = learner.__call__(data)
    for d in data:
       print (classifier(d, orange.GetBoth)),"  ||| ",d
       
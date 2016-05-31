import xml.dom.minidom as dom
#import xml.dom.ext.reader.Sax2
#from xml.dom.ext import PrettyPrint

import os
import sys

class XMLCreator(object):
    def __init__(self):
        self.DOMTreeRoot = None
        self.DOMTreeTop = None


    # attributes is a list od tuples: [(attrName, attrValue), ...]
    def addAttributes(self, node, attributes):
        for attr in attributes:
            node.setAttribute(attr[0], attr[1])


    def insertNewNamedTextNode(self, parent, name, value, attributes=[]):
        new = self.DOMTreeTop.createElement(name)
        new.appendChild(self.DOMTreeTop.createTextNode(value))
        if attributes:
            self.addAttributes(new, attributes)
        parent.appendChild(new)


    def insertNewNode(self, parent, name, attributes=[]):
        new = self.DOMTreeTop.createElement(name)
        if attributes:
            self.addAttributes(new, attributes)
        parent.appendChild(new)
        return new


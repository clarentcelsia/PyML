# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:15:37 2022

@author: user
"""

#OOP
class Animal:
    #self refer to class attr
    def __init__(self, gender, name, age):
        self.gender = gender
        self.name = name
        self.age = age
    
    def toString(self):
        print('%s %s is %d' %(self.gender, self.name, self.age))

#INHERITANCE
class Mammal(Animal):
    #This overrides the parent
    def __init__(self, gender, name, age, skin):
        super().__init__(gender, name, age)
       
        #add child attrs
        self.skin = skin
    
    #*arbitary args == varags/variadic func
    def mammalKids(self, *kids):
        print('%s has 3 kids: %s %s %s' %(self.name, kids[0], kids[1], kids[2]))

    def printMammal(self):
        print('this is mammal: %s %s %s' %(self.skin, self.gender, self.name))

#JSON
    #dumps() : convert python to json.
    #loads() : convert json to python.
#https://www.w3schools.com/python/python_json.asp
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:35:30 2022

@author: user
"""

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def turnRight(self):
        self.x = self.x + 1
        return self.x
    def turnLeft(self):
        self.x = self.x - 1
        return self.x
    def turnUp(self):
        self.y = self.y + 1
        return self.y
    def turnDown(self):
        self.y = self.y - 1
        return self.y
    def currentPosition(self):
        print('current position: (%s, %s)' %(self.x, self.y))


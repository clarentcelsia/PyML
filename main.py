# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:40:10 2022

@author: user
"""

import oop

if __name__=="__main__":
    
    an = oop.Animal('female', 'mochi', 10)
    an.toString()
    #modifying object props
    an.age = 8
    an.toString()
    
    mmal = oop.Mammal('male', 'habo', 12, 'Brown')
    mmal.printMammal()
    mmal.toString()
    mmal.mammalKids('minnie', 'mochi', 'bonnie')
    
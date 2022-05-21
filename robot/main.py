# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:34:22 2022

@author: Allen
"""

import robot

def command(self, direction):
    if direction == 'w':
        robot.Robot.turnLeft(self)
    elif direction == 'e':
        robot.Robot.turnRight(self)
    elif direction == 's':
        robot.Robot.turnDown(self)
    elif direction == 'n':
        robot.Robot.turnUp(self)
    
    robot.Robot.currentPosition(self)

if __name__ == "__main__":
    robocop = input('your starting point (x, y): ')
    myrobot = robot.Robot(
        int(robocop.strip()[0]),
        int(robocop.strip()[2:]))
    q = 'n'
    while q == 'n':
        robodir = input('command: ')
        command(myrobot, robodir)
        q = input('stop(y/n): ')
        if q == 'y':
            q = 'y'
            
        
    

    
    

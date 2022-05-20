# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:35:53 2022

@author: user
"""

#EXERCISE
#1
    #a.Count the number of word in a sentence!
    #b.Count the length of the string
a = 'Banana is yellow'
#strip(): trim whitespace at beginning and end of sentence.
wrd = a.strip().split(' ')
print('a. num of word: ' + str(len(wrd)))
print('b. length of string: ' + str(len(a)))

#2
    #a.Using 'for' to check if the item exists in the list.
    #b.Using the quickest way.
items = list(('stone', 'rock', 'pebbles'))
pick = 'rock'
#a.
def check():
    for item in items:
        print(item) #str
        if item == pick:
            return True
        else:
            pass #just pass no giving error

if check():
    print('true')
else:
    print('false')
    
#b.
if pick in items:
    print('true')
    
#3
    #a.Count the area and volume of tube using oop

#a.
phi = 3.14
class Tube:
    def __init__(self, r, t):
        self.r = r
        self.t = t
    
    def area(self):
        return 2*phi*self.r**2 + 2*phi*self.r*self.t
    
    def volume(self):
        return phi*self.r**2*self.t
    
tube = Tube(4, 2)
area = tube.area()
vol = tube.volume()
print('a tube with {0} m2 of surface of area and {1} of volume'.format(area, vol))


#4. Display the output from given input
my_input = input('insert your birthday (dd/mm/yyyy): ')
date = my_input.strip()[:2]
month = my_input.strip()[4]
year = my_input.strip()[6:]
switcher = {
    1:"January",
    2:"February",
    3:"March",
    4:"April",
    5:"May",
    6:"June",
    7:"July",
    8:"August",
    9:"September",
    10:"October",
    11:"November",
    12:"December"
}

month_str = switcher.get(int(month))
print('Your birthday is on %s %s %s' %(date, month_str, year))
print()

#5. Loop
    #Get the element array of the given input
giv_input = 2
arr = [1,2,4,3]
def count():
    for i in arr:
        if i == giv_input:
            print('element %d is at index %d' %(giv_input, arr.index(giv_input)))
            return
count()

#6. Pattern
#      *
#    *   *
#  *   *   *
row = col = 3
for i in range(row, 0, -1): 
    for j in range(0, i-1, 1): 
      print(" ", end="")
    for j2 in range(col, i-1, -1):
      print("*", end=" ")
    print()

   
        
        
    
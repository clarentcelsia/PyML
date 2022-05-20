# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:57:20 2022

@author: Allen
"""

#VARIABLES
    #Python has no command for declaring a var.
    #Variable will be automatically created the moment you 1st assign a value into it.
    
a = 2
b = "lielly"
print(a) 
print(b)

#CASTING
    #If you want to specify/convert the type of the var, it can be done with Casting.

a = str(1) #--> a = '1'
b = int(3) #--> b = 3
print(a)
print(b)
#nb: 
    #To get the data type, you can use 'type()', i.e type(a) --> <class 'str'>.
    #For string type, you can use either '' or "", both are valid to use.


#MULTIPLE VALUES VARIABLES
    #You can assign multiple value for multiple vars.
        #nb. Python does not have an inbuilt double data type, 
        #but it has a float type that designates a floating-point number.
a, b, c = 1, 'john', float(3.5) 
print(a)
print(b)
print(c)
a = b = c = 'same'
print(a)
print(b)
print(c)


#COLLECTION & OUTPUT
#nb: 
    #List [x, y, z]: Mutable, duplicatable, heterogeneous, ordered
    #Tuple (x, y, z):  Immutable, duplicatable, heterogeneous, ordered
    #Set {x, y, z}: Mutable, unordered, unduplicatable, no indices
    #Frozenset {x,y,z}: Immutable, unordered, unduplicatable, no indices
    #Dict {map(k, v)}: key is unduplicatable
animal = ['butterfly', 'dragon', 'fish']
a = animal[0]
b = animal[1]
c = animal[2]
print(a, b, c) 
    #d = animal[3]
    #print(d) #index out of range
x = y = animal
print(x, y) #--> ['butterfly', 'dragon', 'fish']


#GLOBAL VAR
x = 'this is a global variable'
def yourFunc():
    global x 
    x = 'reassign previous x'
    print('x : ' + x)
    
yourFunc()
print('x after reassigned by global keyword : ' + x)

#DATA TYPES
dStr = 'kelp'                       #TEXT: str
dNum = 22                           #NUMERIC: int, float, complex
dFloat = 22e3                       #e = 10^
dComplex = 1j
dList = [1,2,3,3]                   #SEQUENCE: list (mutable), tuple(immutable) {duplicatable}; range
dTuple = (1,2,3,3)
dRange = range(6) # 0 - 6
dDict = {'name': 'john', 'age': 36} #MAP: dict(ionary) 
dSet = {'1', '2', '3'}              #SET: set (mutable), frozenset (immutable) {unduplicatable}.
dFrozenset = frozenset({'1','2','3'})  
dBool = False                       #BOOL: bool
dBytesInt = bytes(4)                #BINARY: bytes, bytearray, memoryview
dBytesStr = b'hello' 
dByteArray = bytearray(4)
dMemoryView = memoryview(bytes(5))  #print the memory allocation from given obj.
dNone = None                        #NONE: none == null

#STRING
#https://www.w3schools.com/python/python_strings.asp
#STRING, MULTILINE STRINGS
a = '''Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua.'''
print(a)
print(a[1])

#LOOPING
text = 'Lin Qibao'
for char in text:
    print(char) #--> l, i, n, , q, i, ...

#TUPPLE
#Since tupple is immutable, you need to convert it to list type to make it mutable.
#https://www.w3schools.com/python/python_tuples_update.asp

#DICTIONARY
#.keys() & .items()
#https://www.w3schools.com/python/python_dictionaries_loop.asp
print()
cars = {
    "brand":"toyota",
    "name":"CRV",
    "year":"2022",
    "engine":{
        "name":"diesel",
        "year":"2021"
    }
}
print(cars.get("name"))
print(cars.get("engine").get("year"))

cars["color"] = "dope"
cars["engine"].setdefault("brand", "tom")
print(cars)

for attr in cars:
    print(cars[attr])
    

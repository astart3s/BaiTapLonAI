import sys
from sys import argv

first= 0
second=1
# print(len(sys.argv))
if len(sys.argv)==3:
    first = sys.argv[1]
    second = sys.argv[2]
    print(len(sys.argv))
    print("toi la", sys.argv[2])
    print("den tu", sys.argv[1])

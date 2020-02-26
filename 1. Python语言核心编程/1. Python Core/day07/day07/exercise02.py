"""
    4行5列
    *****
    #####
    *****
    #####
"""
for r in range(4):
    for c in range(5):
        if r % 2 !=0:
            print("*",end = "")
        else:
            print("#",end = "")
    print()


for r in range(4):
    for c in range(5):
        if c % 2 !=0:
            print("*",end = "")
        else:
            print("#",end = "")
    print()


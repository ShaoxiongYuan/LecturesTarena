class Person:
    def __init__(self):
        self.勇敢 = True
        self.诚实 = True
        self.善良 = True
        self.英俊 = True


xm = Person()

xm.英俊 = False


class Person:
    def __init__(self):
        self.勇敢 = 1
        self.诚实 = 2
        self.善良 = 4
        self.英俊 = 8

        self.品质 = None


xm = Person()
xm.品质 = 14

if xm.品质 & 善良

xm.品质 = xm.品质 | 勇敢
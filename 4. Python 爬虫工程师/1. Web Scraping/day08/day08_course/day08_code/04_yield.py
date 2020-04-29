def f1():
    for i in range(2):
        yield i

g = f1()
while True:
    try:
        print(next(g))
    except Exception as e:
        break






















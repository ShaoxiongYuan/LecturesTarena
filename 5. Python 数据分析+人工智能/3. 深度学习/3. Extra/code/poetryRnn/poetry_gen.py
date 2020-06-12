# from poetryRnn.poetry_model import PoetryModel
from poetry_model import PoetryModel

if __name__ == '__main__':
    poetry = PoetryModel()
    poem = poetry.gen(poem_len=200)
    print(poem)

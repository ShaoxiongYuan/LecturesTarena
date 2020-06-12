# from avatarDcgan.avatar_model import AvatarModel
from avatar_model import AvatarModel  # wdb 20200325

if __name__ == '__main__':
    avatar = AvatarModel()
    avatar.gen()
    print("图片生成完成.")
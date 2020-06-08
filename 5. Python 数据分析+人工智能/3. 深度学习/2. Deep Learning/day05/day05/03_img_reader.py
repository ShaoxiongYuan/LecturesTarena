# 03_img_reader.py
# 图像样本读取示例
import tensorflow as tf
import os

def img_read(filelist):
    # 构建文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # 定义reader
    reader = tf.WholeFileReader()
    k, v = reader.read(file_queue)
    # 解码
    img = tf.image.decode_jpeg(v)
    # 批处理, 图片需要设置成统一大小
    img_resized = tf.image.resize(img, [200,200])
    img_resized.set_shape([200, 200, 3]) # 设置张量形状

    img_bat = tf.train.batch([img_resized],
                             batch_size=10,
                             num_threads=1)
    return img_bat

if __name__ == "__main__":
    # 创建文件列表
    dir_name = "test_img/"
    file_names = os.listdir(dir_name) # 列出该目录下所有文件
    file_list = []
    for f in file_names:
        # 拼接完整文件路径,并添加到列表
        file_list.append(os.path.join(dir_name, f))
    print(file_list)

    imgs = img_read(file_list)

    with tf.Session() as sess:
        coord = tf.train.Coordinator() # 线程协调器
        # 返回一组线程
        threads = tf.train.start_queue_runners(sess,
                                               coord=coord)

        img_batch = imgs.eval()

        # 等待线层停止并回收资源
        coord.request_stop()
        coord.join(threads)

# 循环显示读取到的图像
import matplotlib.pyplot as plt

plt.figure("Img Show", facecolor="lightgray")

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_batch[i].astype("int32"))

plt.tight_layout() # 紧凑格式
plt.show() # 显示

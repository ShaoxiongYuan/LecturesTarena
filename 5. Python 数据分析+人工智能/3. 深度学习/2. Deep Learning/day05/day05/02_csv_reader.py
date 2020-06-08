# 02_csv_reader.py
# csv文件读取示例
import tensorflow as tf
import os

def csv_read(filelist):
    # 构建文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # 创建csv读取器, 读取数据并解码
    reader = tf.TextLineReader()
    k, v = reader.read(file_queue)
    # 对数据解码
    records = [["None"], ["None"]]
    example, label = tf.decode_csv(v, # 要解码的数据
                            record_defaults=records)# 默认值
    # 批处理(按照制定的笔数凑成批次)
    example_bat, label_bat = tf.train.batch([example, label],
                                            batch_size=4,
                                            num_threads=1)
    return example_bat, label_bat

if __name__ == "__main__":
    # 创建文件列表
    dir_name = "test_data/"
    file_names = os.listdir(dir_name) # 列出该目录下所有文件
    file_list = []
    for f in file_names:
        # 拼接完整文件路径,并添加到列表
        file_list.append(os.path.join(dir_name, f))
    print(file_list)

    example, label = csv_read(file_list)

    with tf.Session() as sess:
        coord = tf.train.Coordinator() # 线程协调器
        # 返回一组线程
        threads = tf.train.start_queue_runners(sess,
                                               coord=coord)
        print(sess.run([example, label])) # 执行文件读取op
        # 等待线层停止并回收资源
        coord.request_stop()
        coord.join(threads)



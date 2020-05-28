import math

# single-pass
import numpy as np
# 以文本在文本集中的顺序列出的文本向量矩阵（用300维向量表示）
text_vec = None
# 以文本在文本集中的顺序列出的话题序号列表
topic_serial = None
# 话题数量
topic_cnt = 0


# single-pass算法
def single_pass(sen_vec, sim_threshold):
    global text_vec
    global topic_serial
    global topic_cnt

    # 向量归一化
    sen_vec = sen_vec * (1.0 / np.linalg.norm(sen_vec))
    if topic_cnt == 0:  # 第1次送入的文本
        # 添加文本向量
        text_vec = sen_vec
        # 话题数量+1
        topic_cnt += 1
        # 分配话题编号，话题编号从1开始
        topic_serial = [topic_cnt]  #第一个进入的，话题编号就是1
    else:  # 第2次及之后送入的文本
        # 文本逐一与已有的话题中的各文本进行相似度计算
        sim_vec = np.dot(sen_vec, text_vec.T)
        # 获取最大相似度值
        max_value = np.max(sim_vec)
        # 获取最大相似度值的文本所对应的话题编号
        topic_ser = topic_serial[np.argmax(sim_vec)]  # np.argmax取出相似度值列表中，值最大的下标.这个下标对应的topic_serial的值，就是所属的类别
        #print("topic_ser", topic_ser, "max_value", max_value)
        # 添加文本向量
        text_vec = np.vstack([text_vec, sen_vec])
        # 分配话题编号
        if max_value >= sim_threshold:
            # 将文本聚合到该最大相似度的话题中
            topic_serial.append(topic_ser)
        else:
            # 否则新建话题，将文本聚合到该话题中
            # 话题数量+1
            topic_cnt += 1
            # 将新增的话题编号（也就是增加话题后的话题数量）分配给当前文本
            topic_serial.append(topic_cnt)

    
# hhh为输入的样本向量实例
hhh = [[1,2],[2,4],[3,6],[4,8],[5,10],[4.5,9],[1,10],[2,20],[3,31],[4,39]]

hhh1 = []

for i in hhh:
    i1 = np.array(i)
    hhh1.append(i1)

hhh2 = np.array(hhh1)

for i in hhh2:
    #print(i)
    single_pass(i, 0.9999)  #第二个参数设置相似度阈值（归一化后，样本点之间的点积）


#topic_serial 每个样本所属类别组成的列表
print(topic_serial)



    

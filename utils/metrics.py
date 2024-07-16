
import numpy as np

'''
scores_i2t：从图像到文本的相似度矩阵。
scores_t2i：从文本到图像的相似度矩阵。
txt2img：字典，表示每个文本对应的图像。
img2txt：字典，表示每个图像对应的文本。
'''
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    '''
    enumerate(scores_i2t) 是 Python 内置函数 enumerate() 的一个用法。
    在这种情况下，scores_i2t 是一个数组或列表，enumerate(scores_i2t) 返回一个迭代器，
    生成一系列的 (index, score) 对，其中 index 是当前元素的索引，score 是 scores_i2t 中对应索引位置的元素。
    '''
    print(scores_i2t.shape)
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]                          #对 score 数组进行排序，并返回排序后的索引，[::-1] 表示降序排列。

        rank = 1e20
        for i in img2txt[index]:                                #遍历当前图像对应的所有文本索引
            tmp = np.where(inds == i)[0][0]                     #找到当前文本索引在排序后的 inds 中的位置
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    print(len(ranks))
    print(ranks)
    print(ranks.shape)
    print(np.where(ranks < 1)[0])
    print(np.where(ranks < 1))
    print(inds.shape)
    print(img2txt[index])
    print(inds)
    # 计算Top-1（tr1）、Top-5（tr5）、Top-10（tr10）的准确率，即在前1、前5、前10个结果中是否有正确的文本。
    tr1 = len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result
import sys
import codecs
import os
import math
import operator
import json
from functools import reduce


def fetch_data(cand, ref):
    """
    @brief：取文本里的数据
    @param：candidate和reference文件的路径
    @return：candidate句子一维列表和reference句子二维列表
    """
    """ Store each reference and candidate sentences as a list """
    references = []
    if '.txt' in ref:
        reference_file = codecs.open(ref, 'r', 'ISO-8859-1')
        references.append(reference_file.readlines())
    else:
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference_file = codecs.open(os.path.join(root, f), 'r', 'ISO-8859-1')
                references.append(reference_file.readlines())
    candidate_file = codecs.open(cand, 'r', 'utf-8')
    candidate = candidate_file.readlines()
    return candidate, references


def count_ngram(candidate, references, n):
    """
    @brief：计算n-gram的得分和简洁惩罚
    @param：candidate句子一维列表和reference句子二维列表
    @return：pr得分, bp简洁惩罚系数
    """
    clipped_count = 0 # 整个文本n-gram较小的那个数量之和
    count = 0 # 整个文本n-gram的数量
    r = 0 # 整个reference文本最匹配长度
    c = 0 # 整个candidate文本全部的单词数量
    # 一句一句处理，当前处理n-gram
    for si in range(len(candidate)):   # []
        FLAG = 1
        # Calculate precision for each sentence
        ref_counts = [] # 存储reference中每个句子中每个n-gram出现的次数
        ref_lengths = [] # 存储reference当前句子的单词数量[number]
        # Build dictionary of ngram counts
        for reference in references: # 对于当前参考，只有一个参考！！
            ref_sentence = reference[si] # 取出参考句子 ref_sentence
            ngram_d = {} # 统计当前句子的每个n-gram出现的次数
            words = ref_sentence.strip().split() # 生成句子的单词列表
            ref_lengths.append(len(words)) # reference当前句子的单词数量
            if len(words) == 0: # 如果是空行，直接跳过
                FLAG = 0
            else:
                FLAG = 1
                limits = len(words) - n + 1 # 计算reference当前句子的n-gram的数量
                # loop through the sentance consider the ngram length
                for i in range(limits):
                    ngram = ' '.join(words[i:i+n]).lower()
                    if ngram in ngram_d.keys():
                        ngram_d[ngram] += 1
                    else:
                        ngram_d[ngram] = 1
                ref_counts.append(ngram_d) # ref_counts是一个列表，里面装了一个字典，该字典里是reference当前句子n-gram数量的统计
        # candidate
        cand_sentence = candidate[si] # 待测句子
        cand_dict = {}
        words = cand_sentence.strip().split()
        if len(words) == 0: # 如果是空行，直接跳过
            FLAG = 0
        else:
            FLAG = 1
            limits = len(words) - n + 1
            for i in range(0, limits): # 统计当前待测句子的每个n-gram出现的次数
                ngram = ' '.join(words[i:i + n]).lower()
                if ngram in cand_dict:
                    cand_dict[ngram] += 1
                else:
                    cand_dict[ngram] = 1
        # cand_dict 是一个字典，里面装了当前candidate句子n-gram数量的统计
        if FLAG == 1:
            clipped_count += clip_count(cand_dict, ref_counts) # 整个文本n-gram较小的那个数量之和
            count += limits # 整个candidate文本n-gram的数量
            r += best_length_match(ref_lengths, len(words)) # 整个reference文本最匹配长度
            c += len(words) # 整个candidate文本全部的单词数量
    if clipped_count == 0: # 如果两个文本完全没相同的n-gram
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """
    @brief：获得限制计数
    @param：cand_d 字典, ref_ds 列表
    @return：count
    """
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys(): # m为candidate字典中键 {}
        m_w = cand_d[m] # 数量
        m_max = 0
        for ref in ref_ds: # 取出字典  []
            if m in ref:  # ref为reference字典
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    # count保存一个句子中n-gram较小的那个数量之和
    return count


def best_length_match(ref_l, cand_l):
    """
    @brief：定义最佳匹配长度
    @param：ref_l, cand_l
    @return：best
    """
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0]) # 当前句子的单词数差
    best = ref_l[0] # reference中当前句子的单词数
    for ref in ref_l:  # 对于当前数量
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    # 对于单个reference文件，就是返回当前reference句子的单词数
    return best


def brevity_penalty(c, r):
    """
    @brief：计算简洁惩罚
    @param：c, r 候选中的数量 参考中的数量
    @return：简洁惩罚系数
    """
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    """
    @brief：几何平均
    @param：数列表
    @return：几何平均值
    """
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(candidate, references, N):
    """
    @brief：计算两个文件的BLEU
    @param：candidate, references 文件路径  N为考虑的n-gram的n
    @return：BLEU得分 0-1
    """
    precisions = []
    # 只考虑1-4 gram
    for i in range(N):
        pr, bp = count_ngram(candidate, references, i+1) # 计算i+1-gram的precision和brevity penalty
        precisions.append(pr)
    # print(precisions)
    bleu = geometric_mean(precisions) * bp # 根据公式计算BLEU
    return bleu


if __name__ == "__main__":

    candidate = "./candidate.txt"
    references = "./testSet/reference.txt"
    
    candidate, references = fetch_data(candidate, references)
    bleu = BLEU(candidate, references, 4) # candidate、reference文件路径、参考几个n-gram
    print(bleu)



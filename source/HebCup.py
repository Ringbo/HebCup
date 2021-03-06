# coding:utf-8

import json
import re
from tqdm import tqdm
from typing import List,Dict
stripAll = re.compile('[\s]+')
from match import match, match_token, equal
from collections import defaultdict
from copy import deepcopy
stop_words = {}
connectOp = {'.', '<con>'}
symbol = {"{","}",":",",","_",".","-","+",";","<con>"}

def lookBack(code_change_seq):
    """
    :param code_change_seq: code change operation sequence
    :return: possible mapping from old comment to new comment
    """
    def itemIsConnect(item):
        if item[0] in connectOp or item[1] in connectOp:
            return True
        else:
            return False

    def combineTuple(mixedTuple):
        res = tuple()
        for x in mixedTuple:
            if isinstance(x, tuple):
                res += x
            else:
                res += tuple((x,))

        if res.__len__() and res[0] in connectOp:
            res = tuple(res[1:])
        if res.__len__() and res[-1] in connectOp:
            res = tuple(res[:-1])
        return res

    def getSubsetMapping(modifiedMapping):
        tempMapping = deepcopy(modifiedMapping)
        for buggyWord in tempMapping:
            for fixedWord in tempMapping[buggyWord]:
                if buggyWord.__len__() == fixedWord.__len__():
                    for i in range(buggyWord.__len__()):
                        for j in range(i + 1, buggyWord.__len__() + 1):
                            if buggyWord[i:j][0] not in connectOp and buggyWord[i:j][-1] not in connectOp \
                                    and fixedWord[i:j][0] not in connectOp and fixedWord[i:j][-1] not in connectOp \
                                    and buggyWord[i:j] != fixedWord[i:j]:
                                modifiedMapping[tuple(buggyWord[i:j])].add(tuple(fixedWord[i:j]))
                else:
                    tempBuggy = list(buggyWord)
                    tempFixed = list(fixedWord)

                    '''
                    Find different part
                                    (pop ->)___________x___(<- pop)
                                    (pop ->)___________xx___(<- pop)
                    '''
                    left_i, left_j, right_i, right_j = 0, 0, tempBuggy.__len__() - 1, tempFixed.__len__() - 1
                    while left_i < tempBuggy.__len__() and left_j < tempFixed.__len__():
                        if tempBuggy[left_i].lower() == tempFixed[left_i].lower():
                            left_i += 1
                            left_j += 1
                        else:
                            left_i = max(0, left_i - 1)
                            left_j = max(0, left_j - 1)
                            break
                    if left_i == tempBuggy.__len__() or left_j == tempFixed.__len__():
                        left_i = max(0, left_i - 1)
                        left_j = max(0, left_j - 1)

                    while right_i >= left_i and right_j >= left_j:
                        if tempBuggy[right_i].lower() == tempFixed[right_j].lower():
                            right_i -= 1
                            right_j -= 1
                        else:
                            right_i += 1
                            right_j += 1
                            break
                    if right_i < 0 or right_j < 0:
                        return modifiedMapping
                    alignedBuggy = tempBuggy[:left_i] + [tuple(tempBuggy[left_i:right_i + 1])] + tempBuggy[right_i + 1:]
                    alignedFixed = tempFixed[:left_j] + [tuple(tempFixed[left_j:right_j + 1])] + tempFixed[right_j + 1:]

                    for i in range(alignedBuggy.__len__()):
                        for j in range(i + 1, alignedFixed.__len__() + 1):
                            key = combineTuple(alignedBuggy[i:j])
                            value = combineTuple(alignedFixed[i:j])
                            if key != value and key.__len__() != 0 and value.__len__() != 0:
                                modifiedMapping[key].add(value)
        return modifiedMapping

    buggyWords = []
    fixedWords = []
    allIndex = []
    lastItem = ['', '', 'equal']
    preHasValidOp = False
    modifiedMapping = defaultdict(set)
    for i, x in enumerate(code_change_seq):
        if x[2] != 'equal':
            allIndex.append(i)
            preHasValidOp = True
        elif (itemIsConnect(lastItem) or itemIsConnect(x)) and preHasValidOp:
            allIndex.append(i)
        else:
            preHasValidOp = False
        lastItem = x

    for i, index in enumerate(allIndex):
        connectFlag = False
        lastItem = code_change_seq[index]
        reversedSeq = list(reversed(code_change_seq[:index]))
        curBuggyWords = []
        curFixedWords = []
        for j, seq in enumerate(reversedSeq):
            if j < index and reversedSeq[j][0] in connectOp or connectFlag:
                curBuggyWords.append(lastItem[0]) if not curBuggyWords.__len__() else None
                curBuggyWords.append(reversedSeq[j][0])
                connectFlag = True

            if j < index and reversedSeq[j][1] in connectOp or connectFlag:
                curFixedWords.append(lastItem[1]) if not curFixedWords.__len__() else None
                curFixedWords.append(reversedSeq[j][1])
                connectFlag = True
            if j < index and reversedSeq[j][0] not in connectOp and reversedSeq[j][1] not in connectOp:
                if connectFlag is False:
                    break
                connectFlag = False
        buggyWords.append(tuple(reversed(tuple(x for x in curBuggyWords if x!=''))))
        fixedWords.append(tuple(reversed(tuple(x for x in curFixedWords if x!=''))))
        if buggyWords[-1].__len__() != 0 and fixedWords[-1].__len__() != 0:
            modifiedMapping[buggyWords[-1]].add(fixedWords[-1])
        if code_change_seq[index][2] == 'replace' and code_change_seq[index][0] not in symbol and code_change_seq[index][1] not in symbol:
            modifiedMapping[tuple((code_change_seq[index][0],))].add(tuple((code_change_seq[index][1],)))

    modifiedMapping = getSubsetMapping(modifiedMapping)
    return modifiedMapping


def getTokenStream(fileInfo):
    """
    Extract infomation stream from preprocessed data file.
    :param fileInfo: Preprocessed data of single file
    :return: old code token stream, new code token stream, old comment token stream, new comment token stream, changed token.
    """
    if "code_change_seq" not in fileInfo:
        return False
    codeSeq = fileInfo["code_change_seq"]
    buggyStream = []
    fixedStream = []
    changed = set()
    for x in codeSeq:
        buggyStream.append(x[0])
        fixedStream.append(x[1])
        if x[2] != "equal":
            changed.add(x[0].lower()) if x[0] != '' and x[0] != '<con>' and x[0].isalpha() and x[0] not in stop_words else None
            changed.add(x[1].lower()) if x[1] != '' and x[1] != '<con>' and x[1].isalpha() and x[1] not in stop_words else None
    buggyStream = [x.lower() for x in buggyStream if x != '' and x !='<con>' and x not in stop_words]
    fixedStream = [x.lower() for x in fixedStream if x != ''and x != '<con>' and x not in stop_words]
    oldComment = [x for x in fileInfo["src_desc_tokens"] if x != '']
    newComment = [x for x in fileInfo["dst_desc_tokens"] if x != '']
    return buggyStream, fixedStream, oldComment, newComment, changed



def sortMapping(streamPair):
    """
    Sort mapping by complexity
    :param streamPair:
    :return:
    """
    modifiedMapping = streamPair[5]
    possibleMapping = []
    for x in modifiedMapping:
        modifiedMapping[x] = list(modifiedMapping[x])
        modifiedMapping[x].sort(key=lambda x:x.__len__(), reverse=True)
        possibleMapping.append((x, modifiedMapping[x]))
    possibleMapping.sort(key=lambda x: x[0].__len__(), reverse=True)
    return possibleMapping


def evaluateCorrectness_mixup(possibleMapping, streamPair, k=1):
    """
    Evaluate the effectiveness of HebCup.
    :param possibleMapping: Possible token mapping from old comment to new comment
    :param streamPair:
    :param k: top k
    :return:
    """
    def genAllpossible(pred):
        allCur = [[]]
        if pred is None:
            return []
        for x in pred:
            tepAllCur = allCur.copy()
            for i in range(allCur.__len__()):
                if isinstance(x, str):
                    tepAllCur[i].append(x)
                elif isinstance(x, list):
                    cur = tepAllCur[i].copy()
                    tepAllCur[i] = None
                    for dst in x:
                        tepAllCur.append(cur + list(dst))
            allCur = [x for x in tepAllCur if x is not None]
        return allCur


    def isEqual_token(pred: List[str], oracle, k):
        if k==1 and pred:
            return Equal_1(pred[0], oracle)
        elif k > 1:
            return Equal_k(pred, oracle, k)
        else:
            return False

    def isEqual(pred, oracle):
        predStr = stripAll.sub(' ', " ".join(pred).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        oracleStr = stripAll.sub(' ', " ".join(oracle).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        predStr = "".join([x for x in predStr if x.isalnum()])
        oracleStr = "".join([x for x in oracleStr if x.isalnum()])
        if predStr.lower() == oracleStr.lower():
            return True
        else:
            return False

    def Equal_1(pred, oracle):
        predStr = "".join(pred).replace("<con>", '')
        oracleStr = "".join(oracle).replace("<con>", '')
        if predStr.lower() == oracleStr.lower():
            return True
        else:
            return False

    def Equal_k(pred: List[str], oracle, k):
        pred.sort(key=lambda x:x.__len__(), reverse=True)
        pred = pred[:k]
        for x in pred:
            if Equal_1(x, oracle):
                return True
        return False

    def split(comment: List[str]):
        comment = " ".join(comment).replace(" <con> ,", " ,").replace(" <con> #", " #").replace(" <con> (", " (") \
            .replace(" <con> )", " )").replace(" <con> {", " {").replace(" <con> }", " }").replace(" <con> @", " @")\
            .replace("# <con> ", "# ").replace(" <con> ", "").strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        return comment.split(" ")

    def tryAllPossible(possibleMapping, streamPair,matchLevel, k):
        cnt = 0
        predComment_token, predComment_subtoken = None, None
        oldComment_token, oldComment_subtoken = None, None
        newComment_token = split(streamPair[3])
        newComment_subtoken = streamPair[3]
        for x in possibleMapping:
            if cnt >= 1:
                break
            if oldComment_token is None:
                oldComment_token = split(streamPair[2])
                oldComment_subtoken = streamPair[2]
            pattern_token = " ".join(x[0]).replace(" <con> ", "").replace(" . ",".")
            pattern_suboten = [x.lower() for x in x[0]]
            pattern_splited = [x.lower() for x in x[0] if x !="<con>"]
            indexes_token = match_token(oldComment_token, pattern_token, matchLevel)
            indexes_subtoken = match(oldComment_subtoken, pattern_suboten, matchLevel)
            indexes_splited = match(oldComment_subtoken, pattern_splited, matchLevel) if pattern_splited else None
            if not indexes_token:
                pass
            else:
                if equal(pattern_token, oldComment_token[indexes_token[0]],1) and not equal(pattern_token,oldComment_token[indexes_token[0]],0):
                    if pattern_token[-1] != 's':
                        x[1][0] = tuple((x[1][0][0] + 's',))
                    else:
                        x[1][0] = tuple((x[1][0][0][:-1],))
                for index in indexes_token:
                    oldComment_token[index] = x[1]
                    predComment_token = oldComment_token
                cnt += 1

            if indexes_subtoken:
                bias = 0
                for index in indexes_subtoken:
                    predComment_subtoken = oldComment_subtoken[:index + bias] + list(x[1][0]) + oldComment_subtoken[index + pattern_suboten.__len__() + bias:]
                    oldComment_subtoken = predComment_subtoken
                    bias = bias + x[1][0].__len__() - x[0].__len__()
                cnt += 1

            if indexes_splited:
                bias = 0
                for index in indexes_splited:
                    predComment_subtoken = oldComment_subtoken[:index + bias] + [y for y in list(x[1][0]) if y != "<con>"] + oldComment_subtoken[index + pattern_splited.__len__() + bias:]
                    oldComment_subtoken = predComment_subtoken
                    bias = bias + x[1][0].__len__() - x[0].__len__()
                cnt += 1

        predComment_token = genAllpossible(predComment_token)

        if predComment_token is not None and isEqual_token(predComment_token, newComment_token, k):
            return True
        elif predComment_subtoken is not None and isEqual(predComment_subtoken, newComment_subtoken):
            return True
        if cnt == 0:
            return None
        else:
            return False

    for i in range(3):
        matchRes = tryAllPossible(possibleMapping, streamPair, matchLevel=i, k=k)
        if matchRes is None:
            continue
        elif matchRes is True:
            return True
        else:
            return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, default="./dataset/test_clean.jsonl", help="dataset_path")
    args = parser.parse_args()
    dataPath = args.dataPath
    allRec = []
    valid = 0
    with open(dataPath, 'r', encoding='utf8') as f:
        for i, x in enumerate(tqdm(f.readlines())):
            fileInfo = json.loads(x)
            buggyStream, fixedStream, src_desc, dst_desc, changed = getTokenStream(fileInfo)
            modifiedMapping = lookBack(fileInfo["code_change_seq"])
            allRec.append([buggyStream, fixedStream, src_desc, dst_desc, changed, modifiedMapping])
    correct = 0
    flags = []
    for i, streamPair in enumerate(allRec):
        possibleMapping = sortMapping(streamPair)
        evalRes_test = evaluateCorrectness_mixup(possibleMapping, streamPair)
        if evalRes_test:
            correct += 1
    print("Accuracy:", correct/allRec.__len__())



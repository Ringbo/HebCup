from tqdm import tqdm
import json
import re
import Levenshtein
from nltk.stem import WordNetLemmatizer
from typing import List,Dict
from tfidf_utils import *
import javalang


stripAll = re.compile("[\s\n\r]+")
stripAllSymbol = re.compile("[~!@#$%^&*()_\+\-\=\[\]\{\}\|;:\'\"<,>.?/]")
# stop_words = set(stopwords.words('english'))  # 1002
# stop_words = {}  # 1004
# connectOp = {'.', '<con>'}

def get_subtokens(method_name):
    returnRec = [x.lower() for x in re.split('[_ \d]',re.sub(r'([A-Z]+[a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', method_name)).strip()) if x !='']
    return returnRec


def oneWordDiff(srcDesc, dstDesc):
    diffs = []
    buggy = []
    fixed = []
    for i in range(srcDesc.__len__()):
        if srcDesc[i].lower() != dstDesc[i].lower():
            diffs.append((srcDesc[i].lower(), dstDesc[i].lower()))
            buggy.append(srcDesc[i].lower())
            fixed.append(dstDesc[i].lower())

    buggy = list(set(buggy))
    fixed = list(set(fixed))

    if buggy.__len__() == fixed.__len__() and (buggy.__len__() == 1 or buggy.__len__() == 0 ):
        return True
    else:
        return False


def camel_case_split(identifier):
    temp = re.sub(r'([A-Z][a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', identifier)).strip().split()
    return [x.lower() for x in temp if x!=""]


def getDiff(srcDesc, dstDesc):
    def onlyOneDiff(diff_0, diff_1):
        if ((set(diff_0) -set(diff_1)).__len__() == 1 and diff_0.__len__() != 1 and diff_0.__len__() - diff_1.__len__() == 1) \
                or( (set(diff_1) -set(diff_0)).__len__() == 1 and diff_1.__len__() != 1 and diff_1.__len__() - diff_0.__len__() == 1):
            return True
        else:
            return False

    def reviseOneDiff(diff_0:List[str],diff_1:List[str]):
        if (set(diff_0) -set(diff_1)).__len__() == 1:
            diff = (set(diff_0) -set(diff_1)).pop()
            diffIndex = diff_0.index(diff)
            if diffIndex == diff_0.__len__() - 1:
                return [(diff_0[-2] + diff_0[-1],diff_1[-1])]
            else:
                return [(diff_0[diffIndex] + diff_0[diffIndex+1], diff_1[diffIndex])]
        else:
            diff = (set(diff_1) - set(diff_0)).pop()
            diffIndex = diff_1.index(diff)
            if diffIndex == diff_1.__len__() -1:
                return [(diff_0[-1], diff_1[-2] + diff_1[-1])]
            else:
                return [(diff_0[diffIndex], diff_1[diffIndex] + diff_1[diffIndex+1])]


    diffs = []
    for i in range(srcDesc.__len__()):
        if srcDesc[i].lower() != dstDesc[i].lower():
            diffs.append((srcDesc[i], dstDesc[i]))
    diffs = list(set(diffs))
    subsetDiffs = []
    tempDiffs = [None, None]
    if not diffs.__len__():
        return diffs
    tempDiffs[0] = camel_case_split(diffs[0][0])
    tempDiffs[1] = camel_case_split(diffs[0][1])
    if tempDiffs[0].__len__() != tempDiffs[1].__len__():
        if not onlyOneDiff(tempDiffs[0], tempDiffs[1]):
            return diffs
        else:
            return reviseOneDiff(tempDiffs[0], tempDiffs[1])

    for i in range(tempDiffs[0].__len__()):
        if tempDiffs[0][i] != tempDiffs[1][i]:
            subsetDiffs.append((tempDiffs[0][i], tempDiffs[1][i]))

    if subsetDiffs.__len__() == 1:
        return subsetDiffs
    else:
        return diffs

wnl = WordNetLemmatizer()

def checkCapitalize(diff):
    if diff.__len__() != 1:
        return False
    # if diff[0][0] != diff[0][1] and diff[0][0].lower() == diff[0][1].lower():
    if diff[0][0].lower() == diff[0][1].lower():
        return True
    else:
        return False

def checkSemantic(diff):
    if diff.__len__() != 1:
        return False

    # Error type of Word Class. e.g., Add -> Adds
    if wnl.lemmatize(diff[0][0], 'n') == wnl.lemmatize(diff[0][1], 'n'):
        return True
    else:
        return False

def checkInclude(diff, possibleWords):
    if diff.__len__() != 1:
        return False
    for x in diff:
        if x[0].lower() in possibleWords and x[1].lower() in possibleWords:
            return True
        buggyWords = set(camel_case_split(x[0]))
        fixedWords = set(camel_case_split(x[1]))
        if (buggyWords - possibleWords).__len__() == 0 and (fixedWords - possibleWords).__len__() == 0:
            return True
        else:
            return False

def checkInclude_Lemmatizer(diff, possibleWords):
    diff = diff.copy()
    # Judge INCLUDE after lemmatizer
    possibleWords = set([wnl.lemmatize(x, 'n') for x in possibleWords])
    diff[0] = (wnl.lemmatize(diff[0][0], 'n'), wnl.lemmatize(diff[0][1], 'n'))
    if diff.__len__() != 1:
        return False
    for x in diff:
        if x[0].lower() in possibleWords and x[1].lower() in possibleWords:
            return True
        buggyWords = set(camel_case_split(x[0]))
        fixedWords = set(camel_case_split(x[1]))
        if (buggyWords - possibleWords).__len__() == 0 and (fixedWords - possibleWords).__len__() == 0:
            return True
        else:
            return False

def checkInsert(srcDesc:List, dstDesc:List):
    srcDesc = srcDesc.copy()
    dstDesc = dstDesc.copy()
    res = [False,0]
    while srcDesc.__len__() and dstDesc.__len__():
        if srcDesc[0] == dstDesc[0]:
            srcDesc.pop(0)
            dstDesc.pop(0)
        else:
            break
    while srcDesc.__len__() and dstDesc.__len__():
        if srcDesc[-1] == dstDesc[-1]:
            srcDesc.pop(-1)
            dstDesc.pop(-1)
        else:
            break
    if abs(srcDesc.__len__() - dstDesc.__len__()) == 1 and (srcDesc.__len__() == 0 or dstDesc.__len__() == 0):
        res[0] = True
        res[1] = srcDesc[0] if srcDesc.__len__() == 1 else dstDesc[0]
    else:
        res[0] = False

    return res


# def checkSinToken(fileInfo):
#     cm_cnt, cd_cnt = 0, 0
#     srcDesc = re.sub(r"[{}@.;,:#()?\-'/\\\]\[\s+]+", ' ', srcDesc).strip()
#
#     return res

def checkSinSubtoken(fileInfo):
    cm_cnt, cd_cnt = 0, 0
    code_change_seq = fileInfo["code_change_seq"]
    desc_change_seq = fileInfo["desc_change_seq"]
    res = {'cm_sin_subtoken': False,
           'cd_sin_subtoken': False,
           'both_sin_subtoken': False,
           }

    for src, dts, op in code_change_seq:
        if op != "equal":
            cd_cnt += 1
    for src, dts, op in desc_change_seq:
        if op != "equal":
            cm_cnt += 1
    if cm_cnt == 1:
        res["cm_sin_subtoken"] = True
    if cd_cnt == 1:
        res["cd_sin_subtoken"] = True
    if cd_cnt == 1 and cm_cnt == 1:
        res["both_sin_subtoken"] = True
    return res

def cntSingleSubtoken(singleSubtokenCnt:Dict, fileInfo):
    res = checkSinSubtoken(fileInfo)
    for x in res:
        if res[x]:
            singleSubtokenCnt[x] += 1

def cntCminCC(fileInfo,possibleWords):
    possibleWords = set([wnl.lemmatize(x, 'n') for x in possibleWords]).copy()
    srcDesc = fileInfo["src_desc_tokens"].copy()
    dstDesc = fileInfo["dst_desc_tokens"].copy()
    while srcDesc.__len__() and dstDesc.__len__():
        if srcDesc[0] == dstDesc[0]:
            srcDesc.pop(0)
            dstDesc.pop(0)
        else:
            break
    while srcDesc.__len__() and dstDesc.__len__():
        if srcDesc[-1] == dstDesc[-1]:
            srcDesc.pop(-1)
            dstDesc.pop(-1)
        else:
            break
    srcDesc = set(srcDesc)
    dstDesc = set(dstDesc)
    changed = (srcDesc | dstDesc) - (srcDesc & dstDesc)
    changed = [wnl.lemmatize(x.lower(), 'n') for x in changed if x.isalpha()]
    for x in changed:
        if x not in possibleWords and x.lower() not in possibleWords:
            return False
    if "".join(changed) in possibleWords:
        return True
    return True

def cntChangeLength(change_seq):
    cnt = 0
    for src, dts, op in change_seq:
        if op != "equal":
            cnt += 1
    return cnt

def getAllTokens(fileInfo):
    src_method = fileInfo['src_method']
    dst_method = fileInfo['dst_method']
    src_comment = fileInfo['src_javadoc']
    dst_comment = fileInfo['dst_javadoc']
    src_cm_tokens = [get_subtokens(x) for x in src_comment.split()]
    dst_cm_tokens = [get_subtokens(x) for x in dst_comment.split()]
    src_tokens = [get_subtokens(x.value) for x in list(javalang.tokenizer.tokenize(src_method))]
    dst_tokens = [get_subtokens(x.value) for x in list(javalang.tokenizer.tokenize(dst_method))]
    src_subtokens = []
    dst_subtokens = []
    src_cm_subtokens = []
    dst_cm_subtokens = []
    for x in src_tokens:
        x = [stripAllSymbol.sub('', t) for t in x]
        src_subtokens += x
    for x in dst_tokens:
        x = [stripAllSymbol.sub('', t) for t in x]
        dst_subtokens += x
    for x in src_cm_tokens:
        x = [stripAllSymbol.sub('', t) for t in x]
        src_cm_subtokens += x
    for x in dst_cm_tokens:
        x = [stripAllSymbol.sub('', t) for t in x]
        dst_cm_subtokens += x
    src_subtokens = [wnl.lemmatize(x, 'n') for x in src_subtokens]
    dst_subtokens = [wnl.lemmatize(x, 'n') for x in dst_subtokens]
    method_subtokens = set(src_subtokens) | set(dst_subtokens)
    cm_diff_subtokens = set(dst_cm_subtokens) - set(src_cm_subtokens)
    return method_subtokens, cm_diff_subtokens


def isCMCInCtx(fileInfo):
    method_subtokens, cm_diff_subtokens = getAllTokens(fileInfo)
    if (cm_diff_subtokens & method_subtokens) == cm_diff_subtokens:
        return True
    else:
        return False


if __name__ == '__main__':
    dataset = 'test'
    dataPath = './dataset/' + dataset + '_clean.jsonl'
    validCnt = 0
    canBeFound = 0
    capitalError = 0
    canbeFixed = []
    cantBeFixed = []
    cmInCC = 0
    singleSubtokenCnt = {"cm_sin_subtoken":0, "cd_sin_subtoken":0, "both_sin_subtoken":0, "can_be_found":0}
    with open(dataPath, 'r', encoding='utf8') as f:
        for i, x in enumerate(tqdm(f.readlines())):
            fileInfo = json.loads(x)
            cntSingleSubtoken(singleSubtokenCnt, fileInfo)
            srcDesc = stripAll.sub(' ', fileInfo["src_desc"])
            # srcDesc = re.sub(r"[{}@.;,:#()?'/\\\-\]\[\s+]+",' ',srcDesc).strip()
            srcDesc = re.sub(r"[{}@.;,:#()?'/\\\]\[\s+]+", ' ', srcDesc).strip()
            dstDesc = stripAll.sub(' ', fileInfo["dst_desc"])
            # dstDesc = re.sub(r"[{}@.;,:#()?'/\\\-\]\[\s]+", ' ', dstDesc).strip()
            dstDesc = re.sub(r"[{}@.;,:#()?'/\\\]\[\s]+", ' ', dstDesc).strip()
            srcDesc = [x for x in srcDesc.split(" ") if x != '']
            dstDesc = [x for x in dstDesc.split(" ") if x != '']
            possibleWords = getPossibleWords(fileInfo)
            if cntCminCC(fileInfo, possibleWords):
                cmInCC += 1
                canbeFixed.append(fileInfo)
            else:
                cantBeFixed.append(fileInfo)
    with open('./dataset/code-indicative.jsonl', 'w', encoding='utf8') as f:
        for x in canbeFixed:
            json.dump(x, f)
            f.write('\n')
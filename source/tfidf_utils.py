
# stop_words = set(stopwords.words('english'))  # 1002
stop_words = {}  # 1004
connectOp = {'.', '<con>'}
symbol = {"{","}",":",",","_",".","-","+",";"}

def lookBack(code_change_seq):
    def removeDup(words):
        temp = set()
        for word in words:
            word = [x.lower() for x in word]
            word.reverse()
            temp.add("|".join(word))

        temp = list(temp)
        temp = [[y for y in x.split('|') if y != ''] for x in temp]  # remove ''
        return [x for x in temp if x.__len__() != 0]

    def itemIsConnect(item):
        if item[0] in connectOp or item[1] in connectOp:
            return True
        else:
            return False

    def getSubset(words):
        subwords = ["".join(word) for word in words]
        for word in words:
            for i in range(len(word)):
                for j in range(i, len(word) + 1):
                    subwords.append("".join(word[i:j]))

        subwords = [x.replace("<con>", '') for x in subwords]
        subwords = [x.strip('.').strip('\"') for x in subwords if x != '' and x not in symbol]
        return set(subwords)

    buggyWords = []
    fixedWords = []
    allIndex = []
    lastItem = ['', '', 'equal']
    preHasValidOp = False
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
        buggyWords.append(curBuggyWords)
        fixedWords.append(curFixedWords)

    buggyWords = removeDup(buggyWords)
    fixedWords = removeDup(fixedWords)

    return getSubset(buggyWords), getSubset(fixedWords)



def getPossibleWords(fileInfo):
    codeSeq = fileInfo["code_change_seq"]
    buggyStream = []
    fixedStream = []
    changed = set()
    for x in codeSeq:
        buggyStream.append(x[0])
        fixedStream.append(x[1])
        if x[2] != "equal":
            changed.add(x[0].lower()) if x[0] != '' and x[0] != '<con>' and x[0].isalpha() and x[
                0] not in stop_words else None
            changed.add(x[1].lower()) if x[1] != '' and x[1] != '<con>' and x[1].isalpha() and x[
                1] not in stop_words else None

    possibleConWords = lookBack(fileInfo["code_change_seq"])

    return changed | possibleConWords[0] | possibleConWords[1]
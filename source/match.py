from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
symbol = set(",.")
def equal(a:str,b:str, level = 0):
    if level == 0 and (a == b or a.lower() == b.lower()):
        return True
    elif level == 1 and (a == b or a + 's' == b or a == b + 's'):
        return True
    elif level == 2 and wnl.lemmatize(a.lower(), 'n') == wnl.lemmatize(b.lower(), 'n'):
        return True
    else:
        return False


def match(string, substring, level=0):
    res = []
    for i, x in enumerate(string):
        if equal(x, substring[0], level):
            j = 0
            for y in substring[1:]:
                if i+j+1 < string.__len__() and equal(y, string[i+j+1],level):
                    j += 1
                    continue
                else:
                    break
            if j == substring.__len__() - 1:
                res.append(i)

    return res

def match_token(string, substring, level=0):
    res = []
    for i, x in enumerate(string):
        if isinstance(x, str) and equal(x, substring, level):
            res.append(i)

    return res



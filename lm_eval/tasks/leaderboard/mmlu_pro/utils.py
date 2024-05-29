import string

def doc_to_text(doc):
    doc_to_text = f"{doc['question']}\n"
    #"A. {{options[0]}}\nB. {{options[1]}}\nC. {{options[2]}}\nD. {{options[3]}}E. {{options[4]}}\nF. {{options[5]}}\nG. {{options[6]}}\nH. {{options[7]}}\nI. {{options[8]}}\nAnswer:"

    for i in range(len(doc["options"])):
        doc_to_text += f"{string.ascii_uppercase[i]}. {doc['options'][i]}\n"

    doc_to_text += "Answer:"
    return doc_to_text


def doc_to_choice(doc):
    return [string.ascii_uppercase[i] for i in range(len(doc["options"]))]

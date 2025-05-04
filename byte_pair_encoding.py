
def initialVocabulary():
    # This is my function to create initial vocabulary.
    
    return list("abcçdefgğhıijklmnoöprsştuüvyzwxq"+
                "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ"+
                "0123456789"+" "+
                "!'^#+$%&/{([)]=}*?\\_-<>|.:´,;`@€¨~\"é")



# This is my function to find most occured pair and bring it.
def bringMostCommonPair(corpus):
    bigram_stats = {}
    for word in corpus:
        for bigram in zip(word, word[1:]):
            bigram_stats[bigram] = bigram_stats.get(bigram, 0) + 1
    sorted_bigrams = sorted(bigram_stats.items(), key=lambda item: (-item[1], item[0]))
    most_common = sorted_bigrams[0][0]
    return (most_common, bigram_stats[most_common])

# This is my function to merge pairs and add to vocabulary.
def merged_tokens(word,pair):
    new_word = []
    new_subword = pair[0] + pair[1]
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(new_subword)
                i += 2
        else:
                new_word.append(word[i])
                i += 1
    return new_word

def bpeCorpus(corpus, maxMergeCount=10):

    # This is my function to create the corpus and vocabulary.
    vocab = initialVocabulary()
    corpus = corpus.split()
    # created a corpus with white spaces and "_"
    for i in range(len(corpus)):
        word = corpus[i]
        corpus[i] = " " + word + "_"
        chars = list(corpus[i])
        corpus[i] = chars

    merges = []

    ####################################################
    for j in range(maxMergeCount):
        # Find the most common pair and add to merges.
        try:
            most_common = bringMostCommonPair(corpus)
        except:
            break
        merges.append(most_common)

        # Update the vocabulary
        new_subword = most_common[0][0] + most_common[0][1]
        vocab.append(new_subword)

        for i in range(len(corpus)):
            word = corpus[i]
            merged = merged_tokens(word, most_common[0])
            corpus[i] = merged


    ####################################################
    tokenized_corpus = corpus

    return merges,vocab,tokenized_corpus# Should return (Merges, Vocabulary, TokenizedCorpus)


def bpeFN(fileName, maxMergeCount=10):
    with open(fileName, 'r',encoding='utf-8') as file:
        corpus = file.read()

    merges,vocab,tokenized_corpus = bpeCorpus(corpus,maxMergeCount)

    return merges,vocab,tokenized_corpus


def bpeTokenize(str, merges):
    corpus = str.split()
    for i in range(len(corpus)):
        word = corpus[i]
        corpus[i] = " " + word + "_"
        chars = list(corpus[i])
        corpus[i] = chars

    for pair in merges:
        for i in range(len(corpus)):
            word = corpus[i]
            nw = merged_tokens(word,pair[0])
            corpus[i] = nw
    return corpus


def bpeFNToFile(infn, maxMergeCount=10, outfn="output.txt"):
    
    # This is my function to create the output file.
    
    (Merges,Vocabulary,TokenizedCorpus)=bpeFN(infn, maxMergeCount)
    outfile = open(outfn,"w",encoding='utf-8')
    outfile.write("Merges:\n")
    outfile.write(str(Merges))
    outfile.write("\n\nVocabulary:\n")
    outfile.write(str(Vocabulary))
    outfile.write("\n\nTokenizedCorpus:\n")
    outfile.write(str(TokenizedCorpus))
    outfile.close()



bpeFNToFile("hw01_bilgisayar.txt", 1000, "hw01-output1.txt")
bpeFNToFile("hw01_bilgisayar.txt", 200, "hw01-output2.txt")





from collections import Counter

def get_inverse_huffman(Huffman):
    invyHuffman = {}
    for i in Huffman.keys():
        invyHuffman[Huffman[i]] = i
    return invyHuffman

def get_stream_to_transmit(Huffman,Encoded):
    BitsToTransmit = []
    for value in Encoded:
        BitsToTransmit.append(Huffman[value])
    return BitsToTransmit

def get_freq_dict(array: list) -> dict:
    """
    returns a dict where the keys are the values of the array, and the values are their frequencies
    :param numpy.ndarray array: intermediary stream as array
    :return: frequency table
    """
    #
    data = Counter(array)
    result = {k: d / len(array) for k, d in data.items()}
    return result


def decode_huffman(cod: str, invHuffman: dict) -> list:
    print("decoding huffman...")
    decodingy = []
    index_init = 0
    for index_fin in range(len(cod)):
        if cod[index_init:index_fin] in invHuffman.keys():
            decodingy.append(invHuffman[cod[index_init:index_fin]])
            index_init = index_fin
    print("complete")
    return decodingy


def find_huffman(p: dict) -> dict:
    """
    returns a Huffman code for an ensemble with distribution p
    :param dict p: frequency table
    :returns: huffman code for each symbol
    """
    p_copy = {}
    p_copy2 = {}
    for i in p.keys():
        p_copy[i] = ""
        p_copy2[str(i)] = ""

    while len(p_copy2) >= 2:
        a1, a2 = lowest_prob_pair(p_copy2)
        p1, p2 = p_copy2.pop(a1), p_copy2.pop(a2)

        for i in a1.split("|"):
            if 'EOB' not in i:
                p_copy[tuple(
                    map(int, i.replace('(', '').replace(')', '').split(', ')))] += "1"
            else:
                p_copy[('EOB',)] += "1"

        for i in a2.split("|"):
            if 'EOB' not in i:
                p_copy[tuple(
                    map(int, i.replace('(', '').replace(')', '').split(', ')))] += "0"
            else:
                p_copy[('EOB',)] += "0"

        p_copy2[a1 + "|" + a2] = p1 + p2

    for i in p_copy.keys():
        p_copy[i] = p_copy[i][::-1]
    return p_copy


def lowest_prob_pair(p):
    # Return pair of symbols from distribution p with lowest probabilities
    sorted_p = sorted(p.items(), key=lambda x: x[1])
    return sorted_p[0][0], sorted_p[1][0]
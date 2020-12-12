import itertools


def flatten(_list):
    return list(itertools.chain(*_list))


def partition(_list, bucket_size):
    return [_list[i:i + bucket_size] for i in range(0, len(_list), bucket_size)]


def normalize_binary_string(binary_string, expected_length):
    return (('0' * (expected_length - len(binary_string))) + binary_string)[-expected_length:]


def sequences(_list):
    prev = _list[0]
    seq = []
    last_i = len(_list) - 2
    for i, item in enumerate(_list[1:]):
        seq.append(prev)
        if i == last_i:
            seq.append(item)
            yield seq
        elif item != prev + 1:
            yield seq
            seq = []
        prev = item
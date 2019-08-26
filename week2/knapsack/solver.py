#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import namedtuple

Item = namedtuple("Item", "idx weight value")

_global_cache = {}
_items = []

def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    input = parse_input(input_data.strip())
    #
    if input['item_count'] <= 200:
        optim, conf, decisions = solve_by_dynamic(input['item_count'], input['capacity'], input['values'],
                                                  input['weights'])
    else:
        optim, conf, decisions = solve_by_raw_value(input['item_count'], input['capacity'], input['values'],
                                                    input['weights'])
    return format_output(optim, conf, decisions)


def solve_by_dynamic(n, k, v, w):
    # Build a list of structs for easy access
    idx = [x for x in range(0, n)]
    # Set up global caches
    global _items
    global _global_cache
    _items = [Item(x[0], x[1], x[2]) for x in zip(idx, w, v)]
    _global_cache.clear()
    # find optimum and backtrack
    optimum = O(k, len(_items)-1)
    decisions = backtrack(k, n-1)
    return optimum, 1, decisions


def backtrack(k, j):
    optimum = _global_cache[k, j]
    decisions = [0] * len(_items)
    while j >= 0:
        if _global_cache[k, j] > _global_cache[k, j-1]:
            decisions[j] = 1
            k = k - _items[j].weight
        j -= 1
    return decisions


def O(k, j):
    global _items
    global _global_cache
    if (k, j) in _global_cache:
        return _global_cache[(k, j)]
    elif j < 0:
        _global_cache[(k, j)] = 0
        return 0
    elif _items[j].weight <= k:
        ret = max(O(k, j-1),
                  _items[j].value + O(k - _items[j].weight, j-1))
        _global_cache[(k, j)] = ret
        return ret
    else:
        ret = O(k, j-1)
        _global_cache[(k, j)] = ret
        return ret



def solve_by_raw_value(n, k, values, weights):
    # Order input by value
    idx = [i for i in range(0, n)]
    indicies = [x for _,_,x in sorted(zip(values, weights, idx), reverse = True)]
   # Init looping variables
    decisions = [0] * n
    optim = 0
    i = 0
    # Logic
    for i in indicies:
        if k <= 0:
            break
        if k - weights[i] > 0:
            k = k - weights[i]
            optim += values[i]
            decisions[i] = 1
    return optim, 0, decisions


def solve_by_order(n, k, values, weights):
    decisions = [0] * n
    optim = 0
    i = 0
    while (i < n and k <= 0):
        if k - weights[i] > 0:
            k = k - weights[i]
            optim += values[i]
            decisions[i] = 1
        i += 1
    return optim, 0, decisions


def format_output(optim, conf, decisions):
    output_data = str(optim) + " " + str(conf) + "\n"
    output_data += ' '.join(map(str, decisions))
    return output_data


def parse_input(input):
    """
    Extract elements of data in to a dictionary
    item_count - int
    capacity - int
    values - list of ints
    weights - list of ints
    """
    # parse the input
    lines = input.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    values = [int(line.split(" ")[0]) for line in lines[1:]]
    weights = [int(line.split(" ")[1]) for line in lines[1:]]

    return {
        "item_count": item_count,
        "capacity": capacity,
        "idx": [i for i in range(0, item_count)],
        "values": values,
        "weights": weights}


def read_input(loc):
    """
    Read a file and return a raw string
    """
    with open(loc, 'r') as input_data_file:
        input_data = input_data_file.read()
    return input_data.strip()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')


## Tests

def test():
    input = get_test_input()
    result = solve_by_dynamic(input['item_count'], input['capacity'], input['values'], input['weights'])
    return result


def get_test_input():
    file = "data/ks_100_0"
    input = parse_input(read_input(file))
    return input

def test_known():
    known = {
        'item_count': 4,
        'capacity': 7,
        'values': [16, 19, 23, 28],
        'weights': [2, 3, 4, 5]}
    result = solve_by_dynamic(
        known['item_count'],
        known['capacity'],
        known['values'],
        known['weights'])
    return result

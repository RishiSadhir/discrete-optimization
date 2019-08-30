from collections import namedtuple

Item = namedtuple("Item", "idx weight value")

_global_cache = {}
_items = []

def solve_it_real(input_data):
    input = parse_input(input_data.strip())

    if input['item_count'] <= 200:
        optim, conf, decisions = solve_by_dynamic(input['item_count'],
                                                  input['capacity'],
                                                  input['values'],
                                                  input['weights'])
    elif input['item_count'] >= 1000:
        optim, conf, decisions = solve_by_density(input['item_count'],
                                                  input['capacity'],
                                                  input['values'],
                                                  input['weights'])
    else:
        optim, conf, decisions = solve_by_density(input['item_count'],
                                                  input['capacity'],
                                                  input['values'],
                                                  input['weights'])
    return format_output(optim, conf, decisions)


def solve_it(input_data):
    print("Solving with lr bab")
    input = parse_input(input_data.strip())
    bab = BranchAndBound(input)
    bab.linear_relaxation_dfs()
    bab.build_decisions()
    return bab.format_output()

def solve_it_dp(input_data):
    input = parse_input(input_data.strip())
    print("Solving with dynamic programming")
    optim, conf, decisions = solve_by_dynamic(input['item_count'],
                                                  input['capacity'],
                                                  input['values'],
                                                  input['weights'])
    return format_output(optim, conf, decisions)

################################
## Branch and Bound
################################

class BranchAndBound:
    global_optimum = 0
    items = []
    items_by_density = []
    decisions = None
    knapsack = ()
    capacity = 0

    def __init__(self, input):
        self.global_optimum = 0
        idx = [x for x in range(0, len(input['values']))]
        self.items = [Item(i, w, v) for i, w, v in
                      zip(idx, input['weights'], input['values'])]
        self.decisions = [0] * len(self.items)
        self.knapsack = ()
        self.capacity = input['capacity']
        self.explored = 0

    def linear_relaxation_dfs(self):
        self._sort_items_by_density()
        self._linear_relaxation_dfs(0, self.capacity, tuple(), 0)
        self.explored = 1

    def _linear_relaxation_dfs(self, value=0, room=None, knapsack = tuple(), i=0):
        if room == None:
            room = self.capacity
        if i >= len(self.items_by_density) or room == 0:
            if value > self.global_optimum:
                self.global_optimum = value
                self.knapsack = knapsack
            return
        elif self._optimistic_evaluation(i, room, value) <= self.global_optimum:
            return
        elif self.items_by_density[i].weight > room:
            self._linear_relaxation_dfs(value, room, knapsack, i+1)
            return
        else:
            self._linear_relaxation_dfs(value, room, knapsack, i+1)
            self._linear_relaxation_dfs(value + self.items_by_density[i].value,
                                        room - self.items_by_density[i].weight,
                                        knapsack + (self.items_by_density[i],),
                                        i+1)

    def _optimistic_evaluation(self, idx, capacity, starting_value = 0):
        optimistic_estimate = starting_value
        for item in self.items_by_density[idx:]:
            if capacity - item.weight > 0:
                capacity -= item.weight
                optimistic_estimate += item.value
            else:
                fraction = capacity / item.weight
                optimistic_estimate += fraction * item.value
                break
        return optimistic_estimate

    def _sort_items_by_density(self):
        densities = [x.value/x.weight for x in self.items]
        self.items_by_density = [item for _,item in
                                 sorted(zip(densities, self.items), reverse = True)]

    def exhaustive_dfs(self):
        self._exhaustive_dfs(0, self.capacity, 0, tuple(), 0)
        self.explored = 1

    def _exhaustive_dfs(self, value, room, estimate, knapsack, i):
        if i >= len(self.items) or room == 0:
            if value > self.global_optimum:
                self.global_optimum = value
                self.knapsack = knapsack
                return
        elif self.items[i].weight > room:
            self._exhaustive_dfs(value, room, estimate, knapsack, i+1)
            return
        else:
            self._exhaustive_dfs(value, room, estimate, knapsack, i+1)
            self._exhaustive_dfs(value + self.items[i].value,
                                 room - self.items[i].weight,
                                 estimate,
                                 knapsack + (self.items[i],),
                                 i+1)

    def build_decisions(self):
        if not self.explored:
            raise ValueError("Havent explored search space.")
        self.decisions = [0] * len(self.items)
        for item in self.knapsack:
            self.decisions[item.idx] = 1

    def format_output(self):
        output_data = str(self.global_optimum) + " " + str(1) + "\n"
        output_data += ' '.join(map(str, self.decisions))
        return output_data

    def __str__(self):
        str = f"Global Optimum: {self.global_optimum}\n"
        str += f"Knapsack: {self.knapsack}\n"
        str += f"Decisions: {self.decisions}"
        return str

    def __repr__(self):
        return str(self)


################################
## Dynamic Programming
################################

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

################################
## Greedy
################################

def solve_by_density(n, k, values, weights):
    idx = [x for x in range(0, n)]
    density = [x[0]/x[1] for x in zip(values, weights)]
    indicies = [x for _,x in sorted(zip(density, idx), reverse = True)]
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

################################
## Utilities
################################

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

################################
## Tests
################################

def test():
    input = get_test_input()
    result = solve_by_dynamic(input['item_count'], input['capacity'], input['values'], input['weights'])
    return result


def get_test_input():
    file = "data/ks_100_0"
    input = parse_input(read_input(file))
    return input

def get_known_input():
    return {
        'item_count': 4,
        'capacity': 7,
        'values': [16, 19, 23, 28],
        'weights': [2, 3, 4, 5]}

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

def test_branch_and_bound():
    input = {
        'item_count': 4,
        'capacity': 7,
        'values': [16, 19, 23, 28],
        'weights': [2, 3, 4, 5]}
    bab = BranchAndBound(input)
    bab.exhaustive_dfs()
    bab.build_decisions()
    return bab

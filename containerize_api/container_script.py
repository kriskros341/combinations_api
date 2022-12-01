from collections import defaultdict
import numpy as np
import json
from typing import List


class Factorial:
    # a class that can operate on factorials that exceed 2^(integer limit) limit
    @classmethod
    def from_value(cls, val):
        roots = defaultdict(lambda: 0)
        for i in range(2, val + 1):
            for x in get_roots_of(i):
                roots[x] += 1
        return cls(roots)

    def __init__(self, roots):
        self.roots = roots

    def __mul__(self, other):
        r = self.roots
        for i, v in other.roots.items():
            r[i] += v
        return Factorial(r)

    def __truediv__(self, other):
        r = self.roots
        for i, v in other.roots.items():
            r[i] -= v
        return Factorial(r)

    def to_value(self):
        res = 1
        for k, v in self.roots.items():
            if v != 0:
                res *= k ** v
        return res


def containerize(containers: List[int], summed: int):

    if any([x == 0 for x in containers]):
        return []

    movement_options = []
    MUST_SUM_UP_TO = summed
    empty_counter = np.array([0] * len(containers))
    def parse_value(value: int, choice1: int, choice2: int):
        # wyraz wartosc za pomocą sum dwóch innych
        current = 0
        counter1 = 0
        counter2 = 0
        while current < value:
            current += choice1
            counter1 += 1

        while (value - current) % choice2 != 0:
            l = (value-current) % choice2
            current -= choice1
            counter1 -= 1
            if counter1 < 0:
                return None

        while current != value:
            current += choice2
            counter2 += 1
        return counter1, counter2


    def pick_min_valid_container():
        for c1 in range(len(containers)-1):
            for c2 in range(c1+1, len(containers)):
                counter = parse_value(MUST_SUM_UP_TO, containers[c1], containers[c2])
                if counter is not None and all(map(lambda x: x >= 0, counter)):
                    t = empty_counter.copy()
                    t[c1] = counter[0]
                    t[c2] = counter[1]
                    return t


    starting_counter = pick_min_valid_container()

    if starting_counter is None:
        return []

    def populate_movement_options():
        #  creates pattern of edges that apply to every vertex
        def increment_on_index_and_create_new(idx: int, counts):
            new_counts = counts.copy()
            new_counts[idx] += 1
            return np.array(new_counts)


        movement_options = []
        path_cache = []

        def helper_fun(current: int, counts, current_idx):
            if current < 0 or current_idx in path_cache:
                return
            if current == 0:
                if sum(counts) != 1:
                    path_cache.append(current_idx)
                    counts = increment_on_index_and_create_new(current_idx, counts*-1)
                    movement_options.append(counts)
                return

            for idx in range(len(containers)):
                helper_fun(current - containers[idx], increment_on_index_and_create_new(idx, counts), current_idx)

        for idx, p in list(enumerate(containers))[1:]:
            helper_fun(p, empty_counter.copy(), idx)
        return movement_options

    def create_graph_iteratively(current, movement_options):
        #  return all the vertices of the graph
        graph = []
        def traverse_iterative(current):
            queue = []
            queue.append(tuple(current))
            graph.append(tuple(current))
            while len(queue) != 0:
                current = queue.pop()
                for movement in movement_options:
                    next_move = tuple(current + movement)
                    if any(x < 0 for x in next_move) or next_move in graph:
                        continue
                    graph.append(next_move)
                    queue.append(next_move)
        traverse_iterative(current)
        return graph


    def create_graph_recursively(current, movement_options):
        #  return all the vertices of the graph
        graph = []

        def traverse_recursive(current):
            if any(x < 0 for x in current):
                return
            temp = tuple(current)
            if temp in graph:
                return
            graph.append(temp)
            for movement in movement_options:
                traverse_recursive(current + movement)
        traverse_recursive(current)
        return graph


    def get_roots_of(val: int):
        #  self explanatory
        roots = []
        while val != 1:
            for x in range(2, val+1):
                if val % x == 0:
                    roots.append(x)
                    val = val // x
                    break
        return roots


    def get_number_of_permutations(graph):
        #  self explanatory
        perm_sum = 0
        for prime_counts in set([tuple(x) for x in graph]):
            j = Factorial.from_value(sum(prime_counts))
            d = Factorial.from_value(0)
            for n in prime_counts:
                d = d * Factorial.from_value(n)
            perm_sum += (j / d).to_value()
        return perm_sum


    def hightest_common_divisor(d1: int, d2: int):
        s1 = set(get_roots_of(d1))
        s2 = set(get_roots_of(d2))
        common = s1.intersection(s2)
        highest_common_divisor = 1 if len(common) == 0 else list(common)[-1]
        return highest_common_divisor


    def lowest_common_multiplication(d1: int, d2: int):
        return d1 * d2 / hightest_common_divisor(d1, d2)


    def create_simple_connection(d1: int, d2: int):
        lcm = lowest_common_multiplication(d1, d2)
        c1 = -lcm / d1
        c2 = lcm / d2
        return c1, c2


    def refine_simple_connection(connection):
        connections_found = []
        if not movement_options:
            # skip checking for first element
            relative_primes = connection / min([abs(x) for x in connection if x != 0])
            if all([x == int(x) for x in relative_primes]):
                movement_options.append(relative_primes)
            else:
                movement_options.append(connection)
            return
        # check if connection already exists
        for option in movement_options:
            new_c = connection - option
            if all(map(lambda x: x % 2 == 0, new_c)):
                new_c = np.array(list(map(lambda x: int(x / 2), new_c)))
            connections_found.append(new_c)
            break # ?!?! so lazzzy

        # make values smallest possible
        relative_primes= connection/min([abs(x) for x in connection if x != 0])

        if all([x == int(x) for x in relative_primes]):
            connections_found.append(relative_primes)
        movement_options.extend(connections_found)

    def connect(idx):
        base = 0
        c = create_simple_connection(containers[base], containers[idx])
        con = empty_counter.copy()
        con[base] = c[0]
        con[idx] = c[1]
        print(con)
        #if con[idx] != 1:
        refine_simple_connection(con)

    for idx in range(1, len(containers)): # one connection has to exist for each node.
        connect(idx)
    movement_options.append(movement_options.pop(0))
    graph = create_graph_iteratively(starting_counter, movement_options)
    return graph


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def stream_containers(containers: List[int], summed: int):

    if any([x == 0 for x in containers]):
        return []

    movement_options = []
    MUST_SUM_UP_TO = summed
    empty_counter = np.array([0] * len(containers))

    def parse_value(value: int, choice1: int, choice2: int):
        # wyraz wartosc za pomocą sum dwóch innych
        current = 0
        counter1 = 0
        counter2 = 0
        while current < value:
            current += choice1
            counter1 += 1

        while (value - current) % choice2 != 0:
            l = (value-current) % choice2
            current -= choice1
            counter1 -= 1
            if counter1 < 0:
                return None

        while current != value:
            current += choice2
            counter2 += 1
        return counter1, counter2


    def pick_min_valid_container():
        for c1 in range(len(containers)-1):
            for c2 in range(c1+1, len(containers)):
                counter = parse_value(MUST_SUM_UP_TO, containers[c1], containers[c2])
                if counter is not None and all(map(lambda x: x >= 0, counter)):
                    t = empty_counter.copy()
                    t[c1] = counter[0]
                    t[c2] = counter[1]
                    return t


    starting_counter = pick_min_valid_container()

    if starting_counter is None:
        return []

    def get_roots_of(val: int):
        #  self explanatory
        roots = []
        while val != 1:
            for x in range(2, val+1):
                if val % x == 0:
                    roots.append(x)
                    val = val // x
                    break
        return roots


    def get_number_of_permutations(graph):
        #  self explanatory
        perm_sum = 0
        for prime_counts in set([tuple(x) for x in graph]):
            j = Factorial.from_value(sum(prime_counts))
            d = Factorial.from_value(0)
            for n in prime_counts:
                d = d * Factorial.from_value(n)
            perm_sum += (j / d).to_value()
        return perm_sum


    def hightest_common_divisor(d1: int, d2: int):
        s1 = set(get_roots_of(d1))
        s2 = set(get_roots_of(d2))
        common = s1.intersection(s2)
        highest_common_divisor = 1 if len(common) == 0 else list(common)[-1]
        return highest_common_divisor


    def lowest_common_multiplication(d1: int, d2: int):
        return d1 * d2 / hightest_common_divisor(d1, d2)


    def create_simple_connection(d1: int, d2: int):
        lcm = lowest_common_multiplication(d1, d2)
        c1 = -lcm / d1
        c2 = lcm / d2
        return c1, c2


    def refine_simple_connection(connection):
        connections_found = []
        if not movement_options:
            # skip checking for first element
            relative_primes = connection / min([abs(x) for x in connection if x != 0])
            if all([x == int(x) for x in relative_primes]):
                movement_options.append(relative_primes)
            else:
                movement_options.append(connection)
            return
        # check if connection already exists
        for option in movement_options:
            new_c = connection - option
            if all(map(lambda x: x % 2 == 0, new_c)):
                new_c = np.array(list(map(lambda x: int(x / 2), new_c)))
            connections_found.append(new_c)
            break # ?!?! so lazzzy

        # make values smallest possible
        relative_primes= connection/min([abs(x) for x in connection if x != 0])

        if all([x == int(x) for x in relative_primes]):
            connections_found.append(relative_primes)
        movement_options.extend(connections_found)

    def connect(idx):
        base = 0
        c = create_simple_connection(containers[base], containers[idx])
        con = empty_counter.copy()
        con[base] = c[0]
        con[idx] = c[1]
        print(con)
        #if con[idx] != 1:
        refine_simple_connection(con)

    for idx in range(1, len(containers)): # one connection has to exist for each node.
        connect(idx)

    def stream_graph_iteratively(current, movement_options):
        #  return all the vertices of the graph
        graph = []

        def traverse_iterative(current):
            queue = []
            queue.append(tuple(current))
            graph.append(tuple(current))
            while len(queue) != 0:

                current = queue.pop()
                for movement in movement_options:
                    next_move = tuple(current + movement)
                    if any(x < 0 for x in next_move) or next_move in graph:
                        continue
                    yield next_move
                    graph.append(next_move)
                    queue.append(next_move)

        yield traverse_iterative(current)

    movement_options.append(movement_options.pop(0))
    gen = stream_graph_iteratively(starting_counter, movement_options)
    return next(gen)



if __name__ == "__main__":
    import sys
    arg = sys.argv
    sums = int(sys.argv[1])
    containers = [int(x) for x in sys.argv[2:]]
    if not sums or not containers:
        print("invalid input")
    else:
        generator = stream_containers(containers, sums)
        for v in generator:
            print(v)

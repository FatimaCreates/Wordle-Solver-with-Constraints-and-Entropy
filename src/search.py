"""Uninformed and informed search over the Wordle belief-state graph.

Every algorithm below searches the SAME graph, so the comparison is fair:

    state      the set of answers still consistent with the clues (belief state)
    action     any guess in the pool
    successor  the belief state after the pattern that guess would produce
    goal       at most one answer left -> we know the hidden word
    cost       one guess per edge, so uniform cost 1
    depth      number of guesses used

Bounded puzzle: a small set of possible answers and a small set of allowed probe
words (defaults: the first 14 answers for both), which keeps the whole space
explorable and the node counts meaningful.  The same machinery is what the full
solver does; here the graph is small enough to be searched exhaustively and
compared.  `depth` counts the guesses needed to IDENTIFY the word; one more guess
then types it in (see run_search.py).

Algorithms
    BFS    breadth-first, complete and optimal for unit costs
    DFS    depth-first, complete here (the graph is acyclic and finite) but not optimal
    UCS    uniform-cost, optimal (identical to BFS with unit costs, shown for the manual)
    DLS    depth-limited DFS with an explicit limit
    GREEDY best-first search: h = the smallest worst-case group size the pool offers
    A*     f = g + h with h = ceil(log_243(k)), admissible because one guess has
           at most 243 outcomes and therefore separates at most 243 * h answers
"""
import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .patterns import N_PATTERNS

METHODS = ("bfs", "dfs", "ucs", "dls", "greedy", "astar")
MAX_OUTCOMES_PER_GUESS = N_PATTERNS          # 243 possible colour patterns


def h_belief(k: int) -> int:
    """Admissible A* heuristic: guesses still needed to separate k answers."""
    if k <= 1:
        return 0
    return int(math.ceil(math.log(k) / math.log(MAX_OUTCOMES_PER_GUESS)))


class BeliefSearch:
    """Graph search over belief states, parameterised by the algorithm used."""

    def __init__(self, pm, pool: Optional[Sequence[str]] = None,
                 answers: Optional[Sequence[str]] = None, max_nodes: int = 20000):
        self.pm = pm
        self.pool = list(pool) if pool else list(pm.answers[:14])
        self.rows = pm.matrix[[pm.guess_index[w] for w in self.pool]]     # (|pool|, |answers|)
        self.words = list(answers) if answers else self.pool             # the hidden words to tell apart
        self.start = np.sort(np.array([pm.answer_index[w] for w in self.words], dtype=np.int64))
        self.max_nodes = int(max_nodes)

    # ---------- graph ----------
    def children(self, cand: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """(guess, belief state) for every guess that actually splits the candidates."""
        out = []
        for word, row in zip(self.pool, self.rows):
            pats = row[cand]
            if pats.min() == pats.max():          # the guess tells us nothing; skip the self-loop
                continue
            for p in np.unique(pats):
                out.append((word, cand[pats == p]))
        return out

    def _h_greedy(self, cand: np.ndarray) -> int:
        """Domain heuristic: the smallest worst-case group size the pool can force."""
        best = len(cand)
        for row in self.rows:
            pats = row[cand]
            if pats.min() == pats.max():
                continue
            counts = np.bincount(pats)
            best = min(best, int(counts.max()))
        return best

    def _priority(self, method: str, cand: np.ndarray, g: int) -> float:
        if method == "ucs":
            return g
        if method == "greedy":
            return min(self._h_greedy(cand), len(cand))
        return g + h_belief(len(cand))              # astar

    # ---------- search ----------
    def search(self, method: str, target: str, depth_limit: int = 3) -> Dict:
        """Run one algorithm for one hidden word. Node counts and path are reported."""
        if method not in METHODS:
            raise ValueError(f"unknown method '{method}', choose from {METHODS}")
        target_index = self.pm.answer_index[target]

        start_key = self.start.tobytes()
        parent: Dict[bytes, Tuple[Optional[bytes], Optional[str]]] = {start_key: (None, None)}
        dist: Dict[bytes, int] = {start_key: 0}
        visited = {start_key}

        if method == "bfs":
            frontier = deque([(self.start, 0)])
        elif method in ("dfs", "dls"):
            frontier = [(self.start, 0)]
        else:
            frontier = []
            heapq.heappush(frontier, (self._priority(method, self.start, 0), 0, self.start, 0))

        expanded = 0
        counter = 0
        while frontier and expanded < self.max_nodes:
            if method == "bfs":
                cand, g = frontier.popleft()
            elif method in ("dfs", "dls"):
                cand, g = frontier.pop()
            else:
                _, _, cand, g = heapq.heappop(frontier)
            key = cand.tobytes()
            if g != dist.get(key):                   # stale priority-queue entry
                continue
            expanded += 1

            # goal: the belief state has narrowed to exactly the hidden word. A group that
            # does not contain the target is a dead end (the real clues can never lead there),
            # and it has no children because the only consistent guess repeats itself.
            if len(cand) == 1 and int(cand[0]) == target_index:
                return self._result(method, target, parent, key, g, expanded)
            if method == "dls" and g >= depth_limit:
                continue

            for word, child in self.children(cand):
                ck = child.tobytes()
                if ck in visited:
                    continue
                visited.add(ck)
                dist[ck] = g + 1
                parent[ck] = (key, word)
                if method in ("bfs", "dfs", "dls"):
                    frontier.append((child, g + 1))
                else:
                    counter += 1
                    heapq.heappush(frontier, (self._priority(method, child, g + 1), counter, child, g + 1))

        return {"method": method, "target": target, "found": False, "depth": None,
                "guesses": [], "expanded": expanded}

    def _result(self, method, target, parent, key, depth, expanded) -> Dict:
        words: List[str] = []
        k = key
        while parent[k][0] is not None:
            k, word = parent[k]
            words.append(word)
        words.reverse()
        return {"method": method, "target": target, "found": True, "depth": depth,
                "guesses": words, "expanded": expanded}


def benchmark(pm, pool: Optional[Sequence[str]] = None, answers: Optional[Sequence[str]] = None,
              methods: Sequence[str] = METHODS, depth_limit: int = 3,
              max_nodes: int = 20000) -> List[Dict]:
    """Every algorithm on every hidden word of a bounded pool. One row per (method, target)."""
    search = BeliefSearch(pm, pool, answers, max_nodes=max_nodes)
    rows = []
    for target in search.words:
        for method in methods:
            r = search.search(method, target, depth_limit=depth_limit)
            rows.append({**r, "pool_size": len(search.pool), "answer_pool_size": len(search.words),
                         "guess_path": " ".join(r["guesses"])})
    return rows

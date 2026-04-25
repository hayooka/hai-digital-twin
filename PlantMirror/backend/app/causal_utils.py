"""
causal_utils.py — loads parents_full.json and walks upstream from a PV.

Graph format: child -> [{parent, lag, level, dynamics, via, lag_method}, ...]
- level 0 : direct DCS link
- level 1 : one physical link hop
- level 2 : two-hop physical causation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Edge:
    child: str
    parent: str
    lag: int
    level: int
    via: str


class CausalGraph:
    def __init__(self, parents: Dict[str, List[dict]]):
        self.parents = parents
        # Inverted index (parent → children) for forward queries
        self.children: Dict[str, List[Edge]] = {}
        for child, plist in parents.items():
            for p in plist:
                e = Edge(
                    child=child,
                    parent=p["parent"],
                    lag=int(p.get("lag", 0)),
                    level=int(p.get("level", 0)),
                    via=str(p.get("via", "")),
                )
                self.children.setdefault(p["parent"], []).append(e)

    @classmethod
    def load(cls, path: Path) -> "CausalGraph":
        with open(path) as f:
            data = json.load(f)
        return cls(data)

    def direct_parents(self, node: str) -> List[Edge]:
        if node not in self.parents:
            return []
        return [
            Edge(
                child=node, parent=p["parent"], lag=int(p.get("lag", 0)),
                level=int(p.get("level", 0)), via=str(p.get("via", "")),
            )
            for p in self.parents[node]
        ]

    def trace_upstream(
        self,
        node: str,
        max_depth: int = 3,
        level_cap: Optional[int] = None,
    ) -> List[Edge]:
        """BFS upstream from `node`. Returns edges in traversal order.

        level_cap: if set, only follow edges with level <= level_cap.
        """
        visited: Set[str] = {node}
        out: List[Edge] = []
        frontier: List[Tuple[str, int]] = [(node, 0)]
        while frontier:
            cur, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for e in self.direct_parents(cur):
                if level_cap is not None and e.level > level_cap:
                    continue
                out.append(e)
                if e.parent not in visited:
                    visited.add(e.parent)
                    frontier.append((e.parent, depth + 1))
        return out

    def rank_suspects(
        self,
        pv: str,
        max_depth: int = 3,
        level_cap: Optional[int] = None,
    ) -> List[Tuple[str, float, List[Edge]]]:
        """Rank upstream sensors by path-length + level weighting.

        Score = sum over each path to `pv` of 1 / (1 + depth + level). Higher is
        more directly implicated.
        """
        scores: Dict[str, float] = {}
        paths: Dict[str, List[Edge]] = {}
        # BFS collecting path-to-root for every ancestor
        frontier: List[Tuple[str, int, List[Edge]]] = [(pv, 0, [])]
        seen: Set[Tuple[str, int]] = set()
        while frontier:
            cur, depth, trail = frontier.pop(0)
            if depth >= max_depth:
                continue
            for e in self.direct_parents(cur):
                if level_cap is not None and e.level > level_cap:
                    continue
                key = (e.parent, depth + 1)
                if key in seen:
                    continue
                seen.add(key)
                weight = 1.0 / (1.0 + depth + e.level)
                scores[e.parent] = scores.get(e.parent, 0.0) + weight
                paths.setdefault(e.parent, trail + [e])
                frontier.append((e.parent, depth + 1, trail + [e]))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(sensor, score, paths[sensor]) for sensor, score in ranked]

import queue
import timeit
import multiprocessing
import math
from joblib import Parallel, delayed
import numpy as np
import sys
from collections import deque,defaultdict
import requests
from pathlib import Path
import json
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import pickle
import os
from dataclasses import dataclass,field
from typing import Tuple,List,Set
import random
import json
from tqdm import tqdm
import argparse
import string
import re
from itertools import product
from datetime import datetime
from geo import *
import networkx as nx

def find_nearest_nodes(node_list, target_coords_list, avg_coords, cos_lat, max_error, role):
    node_positions = np.array([n.proj for n in node_list])
    tree = KDTree(node_positions)

    result = []
    for coord in target_coords_list:
        projected = equirectangular_proj(coord, avg_coords, cos_lat)
        distance, index = tree.query(projected)

        # if distance > max_error:
        #     raise ValueError(
        #         f"{role} coordinate {coord} is not on the map")

        result.append(int(index))
    return result

def compute_distance_matrix(graph: List[List[int]]) -> List[List[int]]:
    n = len(graph)
    dist = [[float("inf")] * n for _ in range(n)]
    for start in range(n):
        queue = deque([(start, 0)])
        visited = set()
        while queue:
            node, d = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            dist[start][node] = d
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, d + 1))
    return dist

def solve(
    graph: List[List[int]],
    pursuers: List[int],
    evaders: List[int],
    max_depth: int = 10
) -> Tuple[str, List[Tuple[List[int], List[int]]]]:
    ESCAPE_NODE = len(graph) - 1
    distance = compute_distance_matrix(graph)
    graph = [neighbors + [i] for i, neighbors in enumerate(graph[:-1])] + [graph[-1]]
    pursuer_graph = [[nbr for nbr in nbrs if nbr!=ESCAPE_NODE] for nbrs in graph]

    def normalize_state(pursuers, evaders):
        return tuple(sorted(pursuers)), tuple(sorted(evaders))

    def pursuer_heuristic(p_moves, evaders):
        scores=[]
        for moves in p_moves:
            total=0
            for i in range(len(moves)):
                dist_to_evaders = max(1e-3, min(distance[moves[i]][e] for e in evaders))
                dist_to_pursuers = min((distance[moves[i]][moves[j]] for j in range(len(moves)) if i!=j),default=1e3)
                total+=-1/dist_to_evaders + 0.5/(dist_to_pursuers+1)
            scores.append(total)

        return [move for _, move in sorted(zip(scores, p_moves))]

    def evader_heuristic(e_moves, pursuers):
        scores=[]
        legal_moves=[]
        for moves in e_moves:
            total=0
            for move in moves:
                dist_from_pursuers = min(distance[move][p] for p in pursuers)
                if dist_from_pursuers==0:
                    break
                toward_escape = distance[move][ESCAPE_NODE]
                if toward_escape==0:
                    return[moves]
                total+=1/dist_from_pursuers -0.5/toward_escape
            else:
                scores.append(total)
                legal_moves.append(moves)

        return [move for _, move in sorted(zip(scores, legal_moves))]

    cache = [None] * 2**20
    def hash_state(state):
        result = 0
        for agent_group in state:
            for a in agent_group:
                result = result * 892973 + a
        return result % len(cache)

    @dataclass
    class CacheEntry:
        depth: int
        result: int
        path: List[Tuple[List[int], List[int]]]
        
        def __post_init__(self):
            self.state_pos = len(self.path)-1

    for depth in range(1, max_depth+1):
        print(f"[{datetime.now()}] Depth {depth}")
        visited = set()
        path = []

        def alpha_beta(depth, alpha, beta, pursuers, evaders, maximizing_player):
            nonlocal path

            caught = set(pursuers) & set(evaders)
            evaders = [e for e in evaders if e not in caught]
            if not evaders:
                path = [(pursuers, evaders)]
                return 1

            if maximizing_player:
                if ESCAPE_NODE in evaders:
                    path = [(pursuers, evaders)]
                    return -1

                state = normalize_state(pursuers, evaders)
                if state in visited:
                    path = [(pursuers, evaders)]
                    return -1
                
                state_hash=hash_state(state)
                cached=cache[state_hash]
                if cached!=None and cached.path[cached.state_pos] == state:
                    if cached.result==0:
                        if cached.depth>=depth:
                            path=[]
                            return cached.result
                    else:
                        path = cached.path[:cached.state_pos+1]
                        return cached.result

            if depth==0:
                path=[]
                return 0

            if maximizing_player: # Pursuers
                visited.add(state)

                pursuer_moves = pursuer_heuristic(list(product(*[pursuer_graph[p] for p in pursuers])),evaders)
                if len(pursuer_moves)==0:
                    path = [(pursuers, evaders)]
                    return -1
                
                score = -2
                for new_positions in pursuer_moves:
                    result = alpha_beta(depth, alpha, beta, list(new_positions), evaders, False)
                    if result > score:
                        score = result
                    alpha = max(alpha, result)
                    if beta <= alpha:
                        break
            
                if score!=0:
                    path.append(state)
                cache[state_hash]=CacheEntry(depth,score,[state] if score==0 else path)
                visited.remove(state)
                
            else: # Evaders
                evader_moves = evader_heuristic(list(product(*[graph[e] for e in evaders])),pursuers)
                if len(evader_moves)==0:
                    path = [(pursuers, evaders)]
                    return -1

                score = 2
                for new_positions in evader_moves:
                    result = alpha_beta(depth - 1, alpha, beta, pursuers, list(new_positions), True)
                    if result < score:
                        score = result
                    beta = min(beta, result)
                    if beta <= alpha:
                        break

            return score

        score = alpha_beta(depth, -1, 1, pursuers, evaders, True)
        if score != 0:
            best_path=list(reversed(path))
            print(score, best_path)
            return score, best_path

    return 0, [(pursuers, evaders)]

def visualize_map(path, nodes, transitions, escape_positions, center, file_name):
    zoom_level = 16

    map_nodes = []
    for key,node in enumerate(nodes):
        map_nodes.append({
            "id": key,
            "lat": node.coords[0],
            "lon": node.coords[1],
            "proj": list(node.proj),
            "color": "#007f00" if key in escape_positions else "#000000"
        })
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Game Map</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <link rel="stylesheet" href="css.css">
    </head>
    <body>
        <div id="top_bar">
            <button onclick="move()">Make move</button>
            <button onclick="undo()">Undo move</button>
            <button onclick="reset()">Reset</button>
            
            <div class="col">
                <label><input type="checkbox" id="auto">Enable Autoplay</label>
                <div>
                    <label for="speed">Autoplay speed:</label>
                    <input type="range" id="speed" min="0" max="100" step="1" value="50">
                </div>
            </div>
            
            <label><input type="checkbox" id="show_trans">Show transitions</label>

            <strong id="eval">Evaluation: —</strong>
        </div>
        <div id="map"></div>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            const map = L.map('map').setView([{center[0]}, {center[1]}], {zoom_level});

            L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 19,
                attribution: '© OpenStreetMap'
            }}).addTo(map);

            const nodes = {json.dumps(map_nodes)};
            const path = {json.dumps(path)};
            const transitions = {json.dumps(transitions)};
        </script>
        <script src="js.js"></script>
    </body>
    </html>
    """

    with open("alphabeta/"+file_name+".html", "w") as map_file:
        map_file.write(html_template)

def main(args):
    nodes,graph,escape_positions,avg_coords,cos_lat=make_map(args.osm_id,args.center,args.radius,args.granularity)
    print(len(graph),"nodes")
    
    # Add terminal state
    for node in range(len(graph)):
        graph[node].append(len(graph))
    graph.append([])

    from collections import Counter
    dist=Counter([len(n) for n in graph])
    print(dict(dist))


    # G = nx.DiGraph()
    # for node, neighbors in enumerate(graph):
    #     for neighbor in neighbors:
    #         G.add_edge(node, neighbor)
    # print("a?",nx.is_planar(G))

    if args.tag==None:
        chars = string.ascii_letters + string.digits  # a-zA-Z0-9
        tag=''.join(random.choices(chars, k=10))
    else:
        if re.search(r"[^A-Za-z0-9_]",args.tag):
            raise ValueError("Tag must be a valid identifier.")
        tag=args.tag

    pursuer_coords = [tuple(p) for p in args.pursuers]
    evader_coords = [tuple(e) for e in args.evaders]

    pursuers = find_nearest_nodes(nodes, pursuer_coords, avg_coords, cos_lat, 2*args.granularity, "Pursuer")
    evaders = find_nearest_nodes(nodes, evader_coords, avg_coords, cos_lat, 2*args.granularity, "Evader")
    _, best_sequence = solve(graph, pursuers, evaders)
    visualize_map(best_sequence, nodes, graph, escape_positions, args.center, tag)

parser = argparse.ArgumentParser()
parser.add_argument('-c','--center', type=float, nargs=2, required=True, help='Coordinates of the center of the game region')
parser.add_argument('-r','--radius', type=float, required=True, help='Radius of the region in meters')
parser.add_argument('-g','--granularity', type=float, default=75, help='Clustering radius of nearby nodes in meters')
parser.add_argument('-p','--pursuers', type=float, action='append', required=True, nargs=2, help='Pursuer coordinates')
parser.add_argument('-e','--evaders', type=float, action='append', required=True, nargs=2, help='Evader coordinates')
parser.add_argument('-o','--osm-id', type=int, default=-1, help='OpenStreetMap relation ID. Obtainable at: https://www.openstreetmap.org/search?query=. -1 or empty to query by center and radius instead.')
# parser.add_argument('-m','--cores', type=int, default=-1, help='Number of CPU cores to use (-1 for all)')
parser.add_argument('-t','--tag', type=str, default=None, help='Tag used to name generated files')
args = parser.parse_args()
main(args)

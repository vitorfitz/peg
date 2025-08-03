import queue
import timeit
import gc
from datetime import datetime
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
from dataclasses import dataclass
import random
import json
import sqlite3
from tqdm import tqdm
import argparse
import string
import re
from math import comb
from itertools import product,combinations_with_replacement
from geo import *

def retrograde(graph, n_pursuers, escape_nodes):
    n_nodes = len(graph)
    adj = [neighbors + [i] for i, neighbors in enumerate(graph)]
    escape_nodes = set(escape_nodes)
    total_combos = comb(n_nodes + n_pursuers - 1, n_pursuers)
    total_positions = total_combos * n_nodes
    print(total_positions,"positions")

    def rank_combo(combo):
        k = len(combo)
        rank = 0
        for i, val in enumerate(combo):
            rank += comb(n_nodes - val + k - i - 2, k - i)
        return total_combos - rank - 1

    def pos_to_index(pos):
        evader, pursuers = pos
        return evader * total_combos + rank_combo(pursuers)
    
    def unrank_combo(rank):
        rank = total_combos - rank - 1
        lex_rank = total_combos - rank - 1
        combo = []
        x = 0
        for i in range(n_pursuers):
            while True:
                c = comb(n_nodes - x + n_pursuers - i - 2, n_pursuers - i - 1)
                if c <= lex_rank:
                    lex_rank -= c
                    x += 1
                else:
                    combo.append(x)
                    break
        return tuple(combo)
    
    def index_to_pos(idx):
        evader = idx // total_combos
        rank = idx % total_combos
        pursuers = unrank_combo(rank)
        return (evader, pursuers)

    results = np.zeros((2, total_positions), dtype=np.int16)
    degree = np.zeros((2, total_positions), dtype=np.int16)
    best_moves = np.full((2, total_positions), -1, dtype=np.int32)

    q = deque()
    idx = 0
    pre_processed=0
    for evader in tqdm(range(n_nodes),desc="Generating terminal states"):
        for pursuers in combinations_with_replacement(range(n_nodes), n_pursuers):
            if evader in pursuers:
                results[0][idx] = 1
                results[1][idx] = 1
                q.append(((evader, pursuers), 0, idx))
                pre_processed+=2
            else:
                degree[1][idx] = len({tuple(sorted(p_moves)) for p_moves in product(*(adj[p] for p in pursuers))})
                if evader in escape_nodes:
                    results[0][idx] = -1
                    q.append(((evader, pursuers), 0, idx))
                    pre_processed+=1
                else:
                    degree[0][idx] = len([a for a in adj[evader] if a not in pursuers])
            idx += 1

    states_to_process=2*total_positions-pre_processed
    pbar = tqdm(total=states_to_process, desc="Retrograde propagation",smoothing=0)
    while q:
        pos, turn, idx = q.popleft()
        val = results[turn][idx]
        evader, pursuers = pos

        if turn == 0:
            # Evader just moved: pursuer's turn
            for p_moves in product(*[adj[p] for p in pursuers]):
                p_sorted = tuple(sorted(p_moves))
                prev_pos = (evader, p_sorted)
                prev_idx = pos_to_index(prev_pos)
                if results[1][prev_idx] == 0:
                    if val > 0:
                        # Pursuers can force capture
                        results[1][prev_idx] = val + 1
                        best_moves[1][prev_idx] = idx
                        pbar.update(1)
                        q.append((prev_pos, 1, prev_idx))
                    elif val < 0:
                        # This move leads to escape, pursuers want to avoid it
                        degree[1][prev_idx] -= 1
                        if degree[1][prev_idx] == 0:
                            # All moves lead to escape
                            results[1][prev_idx] = val - 1  # worst case escape
                            best_moves[1][prev_idx] = idx
                            pbar.update(1)
                            q.append((prev_pos, 1, prev_idx))
        else:
            # Pursuers just moved: evader's turn
            for e_prev in adj[evader]:
                if e_prev in pursuers:
                    continue
                prev_pos = (e_prev, pursuers)
                prev_idx = pos_to_index(prev_pos)
                if results[0][prev_idx] == 0:
                    if val < 0:
                        # Evader can force escape
                        results[0][prev_idx] = val - 1
                        best_moves[0][prev_idx] = idx
                        pbar.update(1)
                        q.append((prev_pos, 0, prev_idx))
                    elif val > 0:
                        # This move leads to capture, evader wants to avoid it
                        degree[0][prev_idx] -= 1
                        if degree[0][prev_idx] == 0:
                            results[0][prev_idx] = val + 1  # worst case capture
                            best_moves[0][prev_idx] = idx
                            pbar.update(1)
                            q.append((prev_pos, 0, prev_idx))

    n_stalemate=states_to_process-pbar.n
    # random non-losing moves for stalemate states
    for turn in [0, 1]:
        for idx in range(total_positions):
            if results[turn][idx] != 0 or best_moves[turn][idx] != -1:
                continue

            pos = index_to_pos(idx)
            evader, pursuers = pos

            if turn == 0:
                # Evader's turn
                for e in adj[evader]:
                    if e in pursuers:
                        continue
                    next_idx = pos_to_index((e, pursuers))
                    if results[1][next_idx] <= 0:  # evader doesn't immediately lose
                        best_moves[0][idx] = next_idx
                        pbar.update(1)
                        break

            else:
                # Pursuers' turn
                for p_moves in product(*[adj[p] for p in pursuers]):
                    p_sorted = tuple(sorted(p_moves))
                    next_idx = pos_to_index((evader, p_sorted))
                    if results[0][next_idx] >= 0:  # pursuers don't immediately lose
                        best_moves[1][idx] = next_idx
                        pbar.update(1)
                        break

    pbar.close()
    tqdm.write(str(n_stalemate)+" stalemate states")
    return results, best_moves, pos_to_index, index_to_pos

def visualize_map(nodes, graph, escape_positions, center, avg_coords, cos_lat, n_pursuers, tag):
    zoom_level = 16

    map_nodes = []
    for key,node in enumerate(nodes):
        map_nodes.append({
            "id": key,
            "lat": node.coords[0],
            "lon": node.coords[1],
            "proj": list(node.proj),
            "color": "#007f00" if key in escape_positions else "#000000",
        })
    
    html_template =\
f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Game Map</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="../static/css.css">
</head>
<body>
    <div id="top_bar">
        <div id="place_agents" class="display_none">
            <span>Place agents:</span>
            <div id="agents"></div>
        </div>
        <div id="top_bar_proper" class="display_none">
            <button onclick="passTurnBtn()">Pass Turn</button>
            <button onclick="undo()">Undo</button>
            <button onclick="reset()">Reset</button>

            <div class="col">
                <label><input type="checkbox" id="ai_pursuer"> AI Pursuers</label>
                <label><input type="checkbox" id="ai_evader"> AI Evaders</label>
            </div>
            
            <div class="col">
                <label for="ai_speed">Speed:</label>
                <input type="range" id="ai_speed" min="0" max="100" step="1" value="50">
            </div>

            <label><input type="checkbox" id="show_trans">Show transitions</label>

            <span id="state_cost" style="margin-left:auto; font-weight:bold;"></span>
        </div>
    </div>
    <div id="map"></div>
    <div id="bottom_left">
        <span id="cursor_coords">Coords: -- --</span>
        <span id="coords_mode">Right click to freeze</span>
    </div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const map = L.map('map').setView([{center[0]}, {center[1]}], {zoom_level});

        L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: 'OpenStreetMap'
        }}).addTo(map);
        
        const degToRad=Math.PI/180;
        function equirectangularProj(lat,lon){{
            return[
                {earth_radius_m}*(lon{(-avg_coords[1]):+})*degToRad*{cos_lat},
                {earth_radius_m}*(lat{(-avg_coords[0]):+})*degToRad
            ];
        }}

        const nodes = {json.dumps(map_nodes)};
        const transitions = {json.dumps(graph)}; // Transitions as list of lists
        const counts = [{n_pursuers}, 1];
        const db="{tag}";
    </script>
    <script src="../static/js2.js"></script>
</body>
</html>
"""

    with open("templates/"+tag+".html", "w") as map_file:
        map_file.write(html_template)

def store_best_moves(results,best_moves,output_dir, tag):
    conn = sqlite3.connect(output_dir+"/"+tag+".db")
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS best_moves")
    cur.execute("CREATE TABLE best_moves (turn INTEGER, state_id INTEGER, best_move INTEGER, cost INTEGER, PRIMARY KEY(turn,state_id))")

    q="INSERT INTO best_moves VALUES (?, ?, ?, ?)"
    pbar=tqdm(desc='Storing moves',smoothing=0,total=2*len(best_moves[0]))

    for turn,moves in enumerate(best_moves):
        batch = []
        for id,move in enumerate(moves):
            batch.append((int(turn), int(id), int(move), int(results[turn][id])))
            if len(batch) >= 10000:
                cur.executemany(q, batch)
                batch.clear()
                pbar.update(10000)

        if batch:
            cur.executemany(q, batch)
            pbar.update(len(batch))
    
    pbar.close()
    conn.commit()
    conn.close()

# @profile
def main(args):
    nodes,graph,escape_positions,avg_coords,cos_lat=make_map(args.osm_id,args.center,args.radius,args.granularity)
    # escape_positions=set()
    print(len(graph),"nodes")

    from collections import Counter
    dist=Counter([len(n) for n in graph])
    print(dict(dist))

    if args.tag==None:
        chars = string.ascii_letters + string.digits  # a-zA-Z0-9
        tag=''.join(random.choices(chars, k=10))
    else:
        if re.search(r"[^A-Za-z0-9_]",args.tag):
            raise ValueError("Tag must be a valid identifier.")
        tag=args.tag

    results,best_moves,_,_ = retrograde(graph, args.pursuers, escape_positions)
    store_best_moves(results,best_moves, args.output_dir, tag)
    visualize_map(nodes, graph, escape_positions, args.center, avg_coords, cos_lat, args.pursuers, tag)

parser = argparse.ArgumentParser()
parser.add_argument('-c','--center', type=float, nargs=2, required=True, help='Latitude and longitude of the center of the game region')
parser.add_argument('-r','--radius', type=float, required=True, help='Radius of the region in meters')
parser.add_argument('-g','--granularity', type=float, default=75, help='Clustering radius of nearby nodes in meters')
parser.add_argument('-p','--pursuers', type=int, required=True, help='Pursuer count')
parser.add_argument('-e','--evaders', type=int, required=True, help='Evader count')
parser.add_argument('-o','--osm-id', type=int, default=-1, help='OpenStreetMap relation ID. Obtainable at: https://www.openstreetmap.org/search?query=. -1 to query by center and radius instead.')
parser.add_argument('-m','--cores', type=int, default=-1, help='Number of CPU cores to use (-1 for all)')
parser.add_argument('-t','--tag', type=str, default=None, help='Tag used to name generated files')
parser.add_argument('-d','--output-dir', type=str, default="./data", help='Where to output db and log files')
args = parser.parse_args()
main(args)

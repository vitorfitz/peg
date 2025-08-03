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
from retrograde_cpp import RetrogradeSolver

def retrograde(graph, n_pursuers, escape_nodes):
    solver = RetrogradeSolver(graph, n_pursuers, escape_nodes)
    solver.run()
    return [solver.get_results(i) for i in [0,1]],[solver.get_best_moves(i) for i in [0,1]]

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

    results,best_moves = retrograde(graph, args.pursuers, escape_positions)
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

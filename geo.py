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

cache_dir="./cache"
Path(cache_dir).mkdir(parents=True, exist_ok=True)
earth_radius_m=6378137
def equirectangular_proj(coords,orig,cos_lat):
    return np.array((
        earth_radius_m*np.radians(coords[1]-orig[1]) * cos_lat,
        earth_radius_m*np.radians(coords[0]-orig[0])
    ))

def inverse_equirectangular_proj(xy, orig, cos_lat):
    x, y          = xy
    lat0, lon0    = orig

    # Recover the angular differences in radians
    dlat_rad =  y / earth_radius_m
    dlon_rad =  x / (earth_radius_m * cos_lat)

    # Convert back to degrees and add to the origin
    lat = lat0 + np.degrees(dlat_rad)
    lon = lon0 + np.degrees(dlon_rad)
    return np.array((lat, lon))

@dataclass
class MapNode:
    coords: Tuple[float, float]
    ways: Set[int] = field(default_factory=set)
    def set_proj(self, orig, cos_lat):
        self.proj=equirectangular_proj(self.coords,orig,cos_lat)

@dataclass
class ClusteredNode:
    proj: Tuple[float, float]
    nodes: List[MapNode]
    ways: Set[int] = None
    def set_coords(self, orig, cos_lat):
        self.coords=inverse_equirectangular_proj(self.proj,orig,cos_lat)

def query_by_relation_id(relation_id):
    pkl_path = os.path.join(cache_dir, f"{relation_id}_graph.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    resp_file=cache_dir+"/"+str(relation_id)+".json"

    try:
        with open(resp_file, 'r') as file:
            data=json.load(file)
    except FileNotFoundError:
        area_id = 3600000000 + relation_id

        # Overpass QL query using an area
        query = f"""
        [out:json];
        area({area_id})->.searchArea;
        way["highway"](area.searchArea);
        (._;>;);
        out body;
        """

        response = requests.post("https://overpass-api.de/api/interpreter", data=query)
        data = response.json()
        with open(resp_file,'w') as file:
            json.dump(data,file)
    
    result=parse_overpass_json(data)
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    return result

def query_by_center_and_radius(center, radius):
    lat, lon = center

    # Overpass QL query using radius
    query = f"""
    [out:json];
    (
      way["highway"](around:{radius},{lat},{lon});
    );
    (._;>;);
    out body;
    """

    response = requests.post("https://overpass-api.de/api/interpreter", data=query)
    response.raise_for_status()  # raise an exception on HTTP errors
    data = response.json()

    return parse_overpass_json(data)

def parse_overpass_json(data):
    nodes = {}
    ways = []
    avg_coords = np.zeros(2)

    for el in data["elements"]:
        if el["type"] == "node":
            coords = np.array((el["lat"], el["lon"]))
            nodes[el["id"]] = MapNode(coords=coords)
            avg_coords += coords
        elif el["type"] == "way" and "nodes" in el:
            ways.append(el)

    avg_coords /= len(nodes)
    cos_lat = np.cos(np.radians(avg_coords[0]))
    for k in nodes:
        nodes[k].set_proj(avg_coords, cos_lat)

    transitions = {n: [] for n in nodes}
    for el in ways:
        way_nodes = el["nodes"]

        for i in range(len(way_nodes) - 1):
            nodes[way_nodes[i]].ways.add(el["id"])
            n1, n2 = way_nodes[i], way_nodes[i + 1]
            if n1 in nodes and n2 in nodes:
                transitions[n1].append(n2)
                transitions[n2].append(n1)  # modify this if one-way handling is added

        nodes[way_nodes[-1]].ways.add(el["id"])

    return nodes, transitions, avg_coords, cos_lat


def graph_dict_to_list(graph_dict):
    # Create a list of unique node labels for consistent indexing
    nodes = graph_dict.keys()
    label_to_index = {label: idx for idx, label in enumerate(nodes)}

    # Create the adjacency list
    adj_list = [[] for _ in nodes]
    for src, neighbors in graph_dict.items():
        src_idx = label_to_index[src]
        adj_list[src_idx] = [label_to_index[dst] for dst in neighbors]

    return adj_list, label_to_index

def circular_clusters(nodes, radius, rebuild_threshold=0.3, max_iters=727):
    random.seed(727)
    values = np.array([n.proj for n in nodes])
    active = np.ones(len(values), dtype=bool)  # Mask for active points
    keys=np.arange(len(values))
    inactive_count=0

    clusters = []
    cluster_map = {}
    tree = KDTree(values)

    while True:
        while True:
            idx = random.randint(0, len(values) - 1)
            if active[idx]:
                break

        center = values[idx]
        cluster=np.array([idx])
        prev_cluster = cluster

        if idx==420:
            print("asd")

        for _ in range(max_iters):
            # Get candidate indices within radius
            candidate_indices = np.array(tree.query_ball_point(center, radius))
            cluster_ways = set()
            for i in cluster:
                cluster_ways.update(nodes[keys[i]].ways)

            # Keep only active candidates that share a way with the cluster
            compatible = []
            for j in candidate_indices:
                if not active[j]:
                    continue
                node_ways = nodes[keys[j]].ways
                if node_ways & cluster_ways:
                    compatible.append(j)
            cluster = np.array(compatible)

            if np.array_equal(cluster, prev_cluster):
                break

            prev_cluster = cluster
            center = np.mean(values[cluster], axis=0)

        # Finalize cluster
        cluster_idx = len(clusters)
        clusters.append(ClusteredNode(
            proj=center,
            nodes=[nodes[keys[j]] for j in cluster],
            # ways=cluster_ways
        ))

        # Update cluster map and deactivate clustered points
        for i in cluster:
            key = keys[i]
            cluster_map[key] = cluster_idx
            active[i] = False

        inactive_count+=len(cluster)

        inactive_ratio = inactive_count / len(values)
        if inactive_ratio==1:
            break

        if inactive_ratio > rebuild_threshold:
            values = values[active]
            keys=keys[active]
            active = np.ones(len(values), dtype=bool)
            tree = KDTree(values)
            inactive_count=0

    return clusters, cluster_map

def build_cluster_graph(graph, cluster_map, n_clusters):
    cluster_graph = [set() for _ in range(n_clusters)]

    for node, neighbors in enumerate(graph):
        c1 = cluster_map[node]
        for nbr in neighbors:
            c2 = cluster_map[nbr]
            if c1 != c2:
                cluster_graph[c1].add(c2)

    return [list(s) for s in cluster_graph]

def remove_long_edges(graph, nodes, max_dist):
    new_graph = [list() for _ in range(len(graph))]

    for node, neighbors in enumerate(graph):
        for nbr in neighbors:
            if np.linalg.norm(nodes[nbr].proj-nodes[node].proj) <= max_dist:
                new_graph[node].append(nbr)

    return new_graph

def prune_graph_by_circle(nodes: dict, adj: dict, center: tuple, radius: float) -> set:
    cx, cy = center
    radius_sq = radius ** 2
    closest_dist_sq=radius_sq

    # Find to_remove and closest_key
    to_remove=set()
    for key, node in nodes.items():
        dist_sq=(node.proj[0] - cx) ** 2 + (node.proj[1] - cy) ** 2
        if dist_sq >= radius_sq:
            to_remove.add(key)
        elif dist_sq < closest_dist_sq:
            closest_dist_sq=dist_sq
            closest_key=key

    # Remove to_remove
    affected = set()
    for key in list(adj):
        prev_n_neighbors = len(adj[key])
        adj[key] = [n for n in adj[key] if n not in to_remove]
        if prev_n_neighbors != len(adj[key]) and key not in to_remove:
            affected.add(key)
    for key in to_remove:
        adj.pop(key, None)
        nodes.pop(key, None)

    if not nodes:
        return affected

    # Find reachable from closest_key
    reachable = set()
    queue = deque([closest_key])
    while queue:
        current = queue.popleft()
        if current in reachable:
            continue
        reachable.add(current)
        for neighbor in adj.get(current, []):
            if neighbor not in reachable:
                queue.append(neighbor)

    # Remove unreachable
    for key in list(nodes):
        if key not in reachable:
            for neighbor in adj.get(key, []):
                if neighbor in adj:
                    adj[neighbor] = [n for n in adj[neighbor] if n != key]
                    affected.add(neighbor)
            adj.pop(key, None)
            nodes.pop(key, None)

    return affected

def split_long_edges(nodes, adjacency, granularity):
    edges_done = set()
    n_neighbors = [len(neighbors) for neighbors in adjacency]

    for i in range(len(adjacency)):
        pos = 0
        while pos < n_neighbors[i]:
            j = adjacency[i][pos]
            if (j, i) in edges_done:
                pos += 1
                continue
            edges_done.add((i, j))

            node1 = nodes[i]
            node2 = nodes[j]

            d = node2.proj - node1.proj
            dist = np.linalg.norm(d)

            num_segments = math.ceil(dist / granularity)
            if num_segments <= 1:
                pos += 1
                continue
            # print(f"Edge {i}, {j}: {dist} -> {dist/num_segments}")
            
            n_neighbors[i]-=1
            n_neighbors[j]-=1
            adjacency[i].remove(j)
            adjacency[j].remove(i)

            shared_ways=node1.ways & node2.ways

            prev_index = i
            for k in range(1, num_segments):
                t = k / num_segments
                nodes.append(ClusteredNode(
                    proj=node1.proj + d*t,
                    nodes=[],
                    ways=shared_ways
                ))

                new_index = len(adjacency)
                adjacency.append([prev_index])
                adjacency[prev_index].append(new_index)
                prev_index = new_index

            # Connect the last intermediate node to j
            adjacency[prev_index].append(j)
            adjacency[j].append(prev_index)

def make_map(relation_id,center,radius,granularity):
    nodes, transitions, avg_coords, cos_lat = \
        query_by_relation_id(relation_id) if relation_id!=-1 \
        else query_by_center_and_radius(center,radius+2*granularity)

    center_proj=equirectangular_proj(center,avg_coords,cos_lat)
    pruned=prune_graph_by_circle(nodes, transitions,center_proj,radius)

    transitions_list,mapping=graph_dict_to_list(transitions)
    nodes_list=[None]*len(mapping)
    for dict_key,list_key in mapping.items():
        nodes_list[list_key]=nodes[dict_key]
    split_long_edges(nodes_list,transitions_list,2*granularity)

    clusters,cluster_map=circular_clusters(nodes_list,granularity)
    # clusters,cluster_map=circular_clusters(nodes_list,0)
    cluster_graph=build_cluster_graph(transitions_list,cluster_map,len(clusters))
    # cluster_graph=remove_long_edges(cluster_graph,clusters,2*granularity)

    for c in clusters:
        c.set_coords(avg_coords,cos_lat)

    cluster_pruned={cluster_map[mapping[p]] for p in pruned if p in mapping and mapping[p] in cluster_map}

    # Add terminal state
    for node in cluster_pruned:
        cluster_graph[node].append(len(cluster_graph))
    cluster_graph.append([])

    return clusters,cluster_graph,cluster_pruned,avg_coords,cos_lat
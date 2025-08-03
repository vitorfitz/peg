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
from typing import Tuple,List,Set
import random
import json
import sqlite3
from tqdm import tqdm
import argparse
import string
import re
from geo import *

def save_partial_state(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_partial_state(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

class Pursuer:
    def __init__(self, velocity):
        self.velocity = velocity

    def setVelocity(self, velocity):
        self.velocity = velocity

    def getVelocity(self):
        return self.velocity


class Evader:
    def __init__(self, velocity):
        self.velocity = velocity
    
    def setVelocity(self, velocity):
        self.velocity = velocity

    def getVelocity(self):
        return self.velocity


class State:
    def __init__(self, id, tuple, turn):
        self.id = id
        self.tuple = tuple
        self.turn = turn
        self.capture = False
        self.cost = 9999990
        self.marked = False
        self.selectedVelocity = 0
        self.transitions = []

    def setId(self, id):
        self.id = id

    def getId(self):
        return self.id

    def setTuple(self, tuple):
        self.tuple = tuple

    def getTuple(self):
        return self.tuple

    def setTurn(self, turn):
        self.turn = turn

    def isTurn(self):
        return self.turn

    def setCapture(self, capture):
        self.capture = capture

    def isCapture(self):
        return self.capture

    def setCost(self, cost):
        self.cost = cost

    def getCost(self):
        return self.cost

    def setMarked(self, marked):
        self.marked = marked

    def isMarked(self):
        return self.marked

    def setSelectedVelocity(self, velocity):
        self.selectedVelocity = velocity

    def getSelectedVelocity(self):
        return self.selectedVelocity

    def setTransitions(self, transitions):
        self.transitions = transitions

    def getTransitions(self):
        return self.transitions


class Transition:
    def __init__(self, origin, destiny, cost):
        self.origin = origin
        self.destiny = destiny
        self.cost = cost

    def setOrigin(self, origin):
        self.origin = origin

    def getOrigin(self):
        return self.origin

    def setDestiny(self, destiny):
        self.destiny = destiny

    def getDestiny(self):
        return self.destiny

    def setCost(self, cost):
        self.cost = cost

    def getCost(self):
        return self.cost


class Temp:
    def __init__(self, tuple):
        self.tuple = tuple
        self.velocity = 0

    def setTuple(self, tuple):
        self.tuple = tuple

    def getTuple(self):
        return self.tuple

    def setVelocity(self, velocity):
        self.velocity = velocity

    def getVelocity(self):
        return self.velocity

global transitionsTurn
transitionsTurn = []

global bases
global reverseBases

bases = {}
reverseBases = {}


def generateBases(pursuers, evaders):
    global lengthSimbol
    global lengthOverhead
    global base
    global agents
    global basesTime
    global reverseBasesTime
    global basesSPD
    global reverseBasesSPD
    global decToHex
    global hexToDec

    lengthOverhead = 1
    agents = len(pursuers) + len(evaders)
    hex = 16
    baseMax = base if base >= time + 2 else time + 2
    exp = 1
    decToHex = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F'}

    hexToDec = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15}

    basesTime = basesSPD = decToHex
    reverseBasesTime = reverseBasesSPD = hexToDec

    while baseMax > hex ** exp:
        exp += 1

    lengthSimbol = exp

    for i in range(baseMax):
        bases[i] = ''

    for i in range(exp):
        threshold = hex ** i
        index = 0
        it = 0
        for j in range(baseMax):
            bases[j] = str(decToHex.get(index)) + bases.get(j)
            it += 1

            if it == threshold:
                index += 1
                it = 0

            if index == hex:
                index = 0

    for i, j in bases.items():
        reverseBases[j] = i

def generatePowers(pursuers, evaders):
    global products
    exp = len(pursuers) + len(evaders) + overHead  # 1 from the bool special pacdot and 1 from time

    products = []

    for i in range(exp):
        products.append([])
        if i < agents:
            pow = base ** i
            for j in range(base):
                products[i].append(j * pow)
        elif i == agents:
            pow = (base) ** i
            for j in range(time + 1):
                products[i].append(j * pow)
        else:
            pow = (time + 1) * base ** agents
            for j in range(2 ** 4):
                products[i].append(j * pow)


def convert(value):
    baseGroup = convertSPD(value[:spd])
    timeID = convertTime(value[spd:overHead])
    tupleID = convertTuple(value[overHead:])
    id = baseGroup + timeID + tupleID
    if id >= traditional:
        id -= gap
    return id


def convertSPD(value):
    return 0 if value=='' else products[agents + 1][convertFromBin(value)]


def convertTime(value):
    return products[agents][int(hexToDec.get(value))]


def convertTuple(value):
    result = 0
    i = 0
    for j in range(len(value) - lengthSimbol, -1, -lengthSimbol):
        result = result + products[i][reverseBases.get(value[j:j + lengthSimbol], 0)]
        i += 1
    return result


def convertToBase(value):
    result = ""
    rest = []

    if value >= traditional:
        value += gap

    i = 0
    timeTuple = value % group
    baseGroup = value // group
    timeID = timeTuple // traditional
    tupleID = timeTuple % traditional  # tupleID

    while True:
        if tupleID < base:
            rest.append(tupleID)
            break
        else:
            temp = tupleID % base
            rest.append(temp)
            tupleID //= base

    for i in range(len(rest)):
        result = bases.get(rest[i], str(rest[i])) + result

    i = len(result)
    while i < agents * lengthSimbol:
        result = '0' + result
        i += 1

    result = decToHex.get(timeID) + result

    result = convertToBin(baseGroup) + result

    i = len(result)
    while i < agents * lengthSimbol + overHead * lengthOverhead:
        result = '0' + result
        i += 1

    return result


def convertToTupleBase(tupleID):
    result = ""
    rest = []
    i = 0

    while True:
        if tupleID < base:
            rest.append(tupleID)
            break
        else:
            temp = tupleID % base
            rest.append(temp)
            tupleID //= base

    for i in range(len(rest)):
        result = bases.get(rest[i], str(rest[i])) + result

    i = len(result)
    while i < agents * lengthSimbol:
        result = '0' + result
        i += 1

    return result


def convertToBin(value):
    result = ""
    rest = []
    i = 0

    value %= traditional

    while True:
        if value < baseBin:
            if value!=0:
                rest.append(value)
            break
        else:
            temp = value % baseBin
            rest.append(temp)
            value //= baseBin

    for i in rest:
        #    result = bases.get(rest[i], str(rest[i])) + result
        result = str(i) + result

    i = len(result)
    while i < spd:
        result = '0' + result
        i += 1

    return result


def convertFromBin(value):
    result = 0
    j = 0
    for i in range(len(value) - 1, -1, -1):
        if value[j] == '1':
            result += 2 ** i
        j += 1

    return result

def isCaptureState(actual, pursuersLength):
    tuplePursuer = actual[0:pursuersLength * lengthSimbol]
    tupleEvader = actual[pursuersLength * lengthSimbol:len(actual)]
    for i in range(0, len(tuplePursuer), lengthSimbol):
        for j in range(0, len(tupleEvader), lengthSimbol):
            if tuplePursuer[i:i + lengthSimbol] == tupleEvader[j:j + lengthSimbol]:
                return True
    return False

def generateStateTransitionsFalse(idTuple, idOffset, spdUnit, t):
    line = []
    eatedPursuer = False
    replace = False

    if spdUnit > 0 and t < time:
        idTuple = linesStateFalse.get(idTuple)
        if enjoyStatesFalse.get(idTuple, -1) != -1:
            if enjoyStatesFalse.get(idTuple) == 9999999:  # enter only if spdUnit != 0 and t < time
                eatedPursuer = True
                replace = True

            tempTuples = queue.Queue(maxsize=0)
            tempPursuers = queue.Queue(maxsize=0)
            tempPursuers.put(Temp(convertToBase(idTuple + idOffset)))
            control = {}
            k = 0
            capture = ''
            for i in range(0, len(pursuers) * lengthSimbol, lengthSimbol):
                while not tempPursuers.empty():
                    tempTuples.put(tempPursuers.get(0))

                while not tempTuples.empty():
                    temp = tempTuples.get()
                    tuple = temp.getTuple()
                    specialDotPart = tuple[0:spd]
                    timePart = tuple[spd:overHead]
                    tuple = tuple[overHead:]
                    firstPart = tuple[0:i]
                    secondPart = tuple[i + lengthSimbol:]
                    pursuersPart = tuple[:len(pursuers) * lengthSimbol]
                    evadersPart = tuple[len(pursuers) * lengthSimbol:]

                    if replace:  # verify if evader capture an evader in special time
                        pPart = pursuersPart
                        for it in range(0, len(pPart), lengthSimbol):
                            if pPart[it:it + lengthSimbol] == evadersPart:
                                capture = pPart[it:it + lengthSimbol]  # position where occurred capture

                        replace = False

                    if i < len(pursuers) * lengthSimbol - lengthSimbol:
                        for velocity in range(pursuers[k].getVelocity()):
                            for j in transitions[int(reverseBases.get(tuple[i:i + lengthSimbol]))][velocity]:
                                if eatedPursuer and capture == pursuersPart[i:i + lengthSimbol]:
                                    newTemp = Temp(specialDotPart + timePart + firstPart + caracters + secondPart)
                                else:
                                    newTemp = Temp(specialDotPart + timePart + firstPart + bases[j] + secondPart)

                                newTemp.setVelocity(velocity if velocity > temp.getVelocity() else temp.getVelocity())
                                tempPursuers.put(newTemp)

                    else:
                        for velocity in range(pursuers[k].getVelocity()):
                            for j in transitions[reverseBases.get(tuple[i:i + lengthSimbol])][velocity]:
                                if (spdUnit == 0 or spdUnit != 0 and t == time) and timeGroup != 1:
                                    if evadersPart in specialPacDot:  # if enter, transition leads to other group
                                        for bit in range(len(specialPacDot)):
                                            if specialDotPart[bit] == '0' and evadersPart == specialPacDot[
                                                bit]:  # if evader in pac-dot especial position
                                                if eatedPursuer and capture == pursuersPart[
                                                                               i:i + lengthSimbol]:  # if evader penalizes pursuer
                                                    newTuple = Temp(specialTransitions[spdUnit][
                                                                        bit] + '0' + firstPart + caracters + secondPart)  # time == 0
                                                else:
                                                    newTuple = Temp(
                                                        specialTransitions[spdUnit][bit] + '0' + firstPart + bases[
                                                            j] + secondPart)  # time == 0
                                                break
                                            else:  # maintain transition in the same group and time, that is, with out special time
                                                newTuple = Temp(
                                                    specialDotPart + timePart + firstPart + bases[j] + secondPart)
                                    else:  # maintain transition in the same group and time, that is, with out special time
                                        newTuple = Temp(specialDotPart + timePart + firstPart + bases[j] + secondPart)
                                else:
                                    if timeGroup == 1:
                                        newTuple = Temp(spd * '0' + '0' + firstPart + bases[j] + secondPart)
                                    else:
                                        if eatedPursuer and capture == pursuersPart[i:i + lengthSimbol]:
                                            newTuple = Temp(specialDotPart + decToHex.get(
                                                hexToDec.get(timePart) + 1) + firstPart + caracters + secondPart)
                                        else:
                                            newTuple = Temp(
                                                specialDotPart + decToHex.get(hexToDec.get(timePart) + 1) + firstPart +
                                                bases[j] + secondPart)

                                if eatedPursuer:
                                    if caracters in newTuple.getTuple():
                                        newTuple.setTuple(
                                            newTuple.getTuple().replace(caracters, bases.get(int(reborn[0][0]))
                                            if not bases.get(int(reborn[0][0])) in evadersPart else bases.get(
                                                int(reborn[0][1]))))

                                newTuple.setVelocity(velocity if velocity > temp.getVelocity() else temp.getVelocity())
                                tempPursuers.put(newTuple)
                                id = convert(newTuple.getTuple())
                                if control.get(id) == None:
                                    line.append(id)
                                control[id] = 0
                k += 1
    else:
        if statesFalse[0][idTuple] > 0:
            if statesFalse[0][idTuple] == 9999999:  # enter only if spdUnit != 0 and t < time
                eatedPursuer = True
                replace = True

            tempTuples = queue.Queue(maxsize=0)
            tempPursuers = queue.Queue(maxsize=0)
            tempPursuers.put(Temp(convertToBase(idTuple + idOffset)))
            control = {}
            k = 0
            capture = ''
            for i in range(0, len(pursuers) * lengthSimbol, lengthSimbol):
                while not tempPursuers.empty():
                    tempTuples.put(tempPursuers.get(0))

                while not tempTuples.empty():
                    temp = tempTuples.get()
                    tuple = temp.getTuple()
                    specialDotPart = tuple[0:spd]
                    timePart = tuple[spd:overHead]
                    tuple = tuple[overHead:]
                    firstPart = tuple[0:i]
                    secondPart = tuple[i + lengthSimbol:]
                    pursuersPart = tuple[:len(pursuers) * lengthSimbol]
                    evadersPart = tuple[len(pursuers) * lengthSimbol:]
                    # """
                    if replace:  # verify if evader capture an evader in special time
                        pPart = pursuersPart
                        for it in range(0, len(pPart), lengthSimbol):
                            if pPart[it:it + lengthSimbol] == evadersPart:
                                capture = pPart[it:it + lengthSimbol]  # position where occurred capture

                        replace = False
                    if i < len(pursuers) * lengthSimbol - lengthSimbol:
                        for velocity in range(pursuers[k].getVelocity()):
                            for j in transitions[int(reverseBases.get(tuple[i:i + lengthSimbol]))][velocity]:
                                if eatedPursuer and capture == pursuersPart[i:i + lengthSimbol]:
                                    newTemp = Temp(specialDotPart + timePart + firstPart + caracters + secondPart)
                                else:
                                    newTemp = Temp(specialDotPart + timePart + firstPart + bases[j] + secondPart)

                                newTemp.setVelocity(velocity if velocity > temp.getVelocity() else temp.getVelocity())
                                tempPursuers.put(newTemp)

                    else:
                        for velocity in range(pursuers[k].getVelocity()):
                            for j in transitions[reverseBases.get(tuple[i:i + lengthSimbol])][velocity]:
                                if (spdUnit == 0 or spdUnit != 0 and t == time) and timeGroup != 1:
                                    if evadersPart in specialPacDot:  # if enter, transition leads to other group
                                        for bit in range(len(specialPacDot)):
                                            if specialDotPart[bit] == '0' and evadersPart == specialPacDot[
                                                bit]:  # if evader in pac-dot especial position
                                                if eatedPursuer and capture == pursuersPart[
                                                                               i:i + lengthSimbol]:  # if evader penalizes pursuer
                                                    newTuple = Temp(specialTransitions[spdUnit][
                                                                        bit] + '0' + firstPart + caracters + secondPart)  # time == 0
                                                else:
                                                    newTuple = Temp(
                                                        specialTransitions[spdUnit][bit] + '0' + firstPart + bases[
                                                            j] + secondPart)  # time == 0
                                                break
                                            else:  # maintain transition in the same group and time, that is, with out special time
                                                newTuple = Temp(
                                                    specialDotPart + timePart + firstPart + bases[j] + secondPart)
                                    else:  # maintain transition in the same group and time, that is, with out special time
                                        newTuple = Temp(specialDotPart + timePart + firstPart + bases[j] + secondPart)
                                else:

                                    if timeGroup == 1:
                                        newTuple = Temp(spd * '0' + '0' + firstPart + bases[j] + secondPart)
                                    else:
                                        if eatedPursuer and capture == pursuersPart[i:i + lengthSimbol]:
                                            newTuple = Temp(specialDotPart + decToHex.get(
                                                hexToDec.get(timePart) + 1) + firstPart + caracters + secondPart)
                                        else:
                                            newTuple = Temp(
                                                specialDotPart + decToHex.get(hexToDec.get(timePart) + 1) + firstPart +
                                                bases[j] + secondPart)
                                
                                if eatedPursuer:
                                    if caracters in newTuple.getTuple():
                                        newTuple.setTuple(newTuple.getTuple().replace(caracters, bases.get(
                                            int(reborn[0][0])) if not bases.get(
                                            int(reborn[0][0])) in evadersPart else bases.get(int(reborn[0][1]))))

                                newTuple.setVelocity(velocity if velocity > temp.getVelocity() else temp.getVelocity())
                                tempPursuers.put(newTuple)
                                id = convert(newTuple.getTuple())
                                if control.get(id) == None:
                                    line.append(id)
                                control[id] = 0
                k += 1

    return line

def generateStateTransitionsTrue(idTuple, idOffset, spdUnit, t):
    line = []
    replace = False

    if spdUnit > 0 and t < time:
        idTuple = linesStateTrue.get(idTuple)
        if enjoyStatesTrue.get(idTuple, -1) != -1:
            if enjoyStatesTrue.get(idTuple) == 9999999:  # enter only if spdUnit != 0 and t < time
                replace = True

            tempTuples = queue.Queue(maxsize=0)
            tempPursuers = queue.Queue(maxsize=0)
            tempPursuers.put(Temp(convertToBase(idTuple + idOffset)))
            control = {}
            i = len(pursuers) * lengthSimbol
            k = len(pursuers)
            while i < (len(pursuers) + len(evaders)) * lengthSimbol:
                while not tempPursuers.empty():
                    tempTuples.put(tempPursuers.get(0))

                while not tempTuples.empty():
                    temp = tempTuples.get()
                    tuple = temp.getTuple()
                    specialDotPart = tuple[0:spd]
                    timePart = tuple[spd:overHead]
                    tuple = tuple[overHead:]
                    firstPart = tuple[0:i]
                    secondPart = tuple[i + lengthSimbol:]
                    pursuersPart = tuple[:len(pursuers) * lengthSimbol]
                    evadersPart = tuple[len(pursuers) * lengthSimbol:]
                    if replace:
                        pPart = pursuersPart
                        for it in range(0, len(pPart), lengthSimbol):
                            if pPart[
                               it:it + lengthSimbol] == evadersPart:  # if evader capture a pursuer, the pursuer's capture position is changed here
                                pursuersPart = pursuersPart.replace(pPart[it:it + lengthSimbol],
                                                                    bases.get(int(reborn[0][0])) if not bases.get(int(
                                                                        reborn[0][0])) == evadersPart else bases.get(
                                                                        int(reborn[0][1])))

                        replace = False
                        tuple = pursuersPart + evadersPart
                        firstPart = tuple[0:i]
                        secondPart = tuple[i + lengthSimbol:]
                        pursuersPart = tuple[:len(pursuers) * lengthSimbol]

                    if i < (len(pursuers) + len(evaders)) * lengthSimbol - lengthSimbol:
                        for velocity in range(evaders[k - len(pursuers)].getVelocity()):
                            for j in transitions[int(reverseBases.get(tuple[i:i + lengthSimbol]))][velocity]:
                                newTemp = Temp(specialDotPart + timePart + firstPart + bases[j] + secondPart)
                                newTemp.setVelocity(velocity if velocity > temp.getVelocity() else temp.getVelocity())
                                tempPursuers.put(newTemp)

                    else:
                        for velocity in range(evaders[k - len(pursuers)].getVelocity()):
                            for j in transitions[int(reverseBases.get(tuple[i:i + lengthSimbol]))][velocity]:
                                newTuple = Temp(specialDotPart + timePart + firstPart + bases[j] + secondPart)
                                if not isFeasibleCompleteState(newTuple.getTuple(), spdUnit, t):
                                    continue

                                newTuple.setVelocity(velocity if velocity > temp.getVelocity() else temp.getVelocity())
                                tempPursuers.put(newTuple)
                                id = convert(newTuple.getTuple())
                                if control.get(id) == None:
                                    line.append(id)
                                    control[id] = 0
                i += lengthSimbol
                k += 1

    else:
        if statesTrue[0][idTuple] > 0:
            if statesTrue[0][idTuple] == 9999999:  # enter only if spdUnit != 0 and t < time
                replace = True

            tempTuples = queue.Queue(maxsize=0)
            tempPursuers = queue.Queue(maxsize=0)
            tempPursuers.put(Temp(convertToBase(idTuple + idOffset)))
            control = {}
            i = len(pursuers) * lengthSimbol
            k = len(pursuers)
            while i < (len(pursuers) + len(evaders)) * lengthSimbol:
                while not tempPursuers.empty():
                    tempTuples.put(tempPursuers.get(0))

                while not tempTuples.empty():
                    temp = tempTuples.get()
                    tuple = temp.getTuple()
                    specialDotPart = tuple[0:spd]
                    timePart = tuple[spd:overHead]
                    tuple = tuple[overHead:]
                    if evaders[0].getVelocity() > 1:
                        transitions2 = generateAllEvaderTransitions(transitions, evaders[0].getVelocity())
                    else:
                        transitions2 = transitions

                    firstPart = tuple[0:i]
                    secondPart = tuple[i + lengthSimbol:]
                    pursuersPart = tuple[:len(pursuers) * lengthSimbol]
                    evadersPart = tuple[len(pursuers) * lengthSimbol:]
                    if replace:
                        pPart = pursuersPart
                        for it in range(0, len(pPart), lengthSimbol):
                            if pPart[
                               it:it + lengthSimbol] == evadersPart:  # if evader capture a pursuer, the pursuer's capture position is changed here
                                pursuersPart = pursuersPart.replace(pPart[it:it + lengthSimbol],
                                                                    bases.get(int(reborn[0][0])) if not bases.get(int(
                                                                        reborn[0][0])) == evadersPart else bases.get(
                                                                        int(reborn[0][1])))

                        replace = False
                        tuple = pursuersPart + evadersPart
                        firstPart = tuple[0:i]
                        secondPart = tuple[i + lengthSimbol:]
                        pursuersPart = tuple[:len(pursuers) * lengthSimbol]

                    if i < (len(pursuers) + len(evaders)) * lengthSimbol - lengthSimbol:
                        for velocity in range(evaders[k - len(pursuers)].getVelocity()):
                            for j in transitions2[int(reverseBases.get(tuple[i:i + lengthSimbol]))][velocity]:
                                newTemp = Temp(specialDotPart + timePart + firstPart + bases[j] + secondPart)
                                newTemp.setVelocity(velocity if velocity > temp.getVelocity() else temp.getVelocity())
                                tempPursuers.put(newTemp)

                    else:
                        for velocity in range(evaders[k - len(pursuers)].getVelocity()):
                            for j in transitions2[int(reverseBases.get(tuple[i:i + lengthSimbol]))][velocity]:
                                if not isCaptureState(firstPart + bases[j] + secondPart, len(pursuers)):
                                    newTuple = Temp(specialDotPart + timePart + firstPart + bases[j] + secondPart)

                                    # if not(eatedPursuer or enter) or (eatedPursuer and enter):
                                    newTuple.setVelocity(velocity if velocity > temp.getVelocity() else temp.getVelocity())
                                    tempPursuers.put(newTuple)
                                    id = convert(newTuple.getTuple())
                                    if control.get(id) == None:
                                        line.append(id)
                                        control[id] = 0
                i += lengthSimbol
                k += 1

    return line

def pathsFromPacDots(transitions, specialPacDot):
    max_hops = time
    global evadersPaths
    evadersPaths = []

    for k in range(len(specialPacDot)):
        pd = reverseBases.get(specialPacDot[k])  # Start node index
        evadersPaths.append([])
        visited = {pd: 0}  # Mark starting node as visited

        for hop in range(max_hops):
            evadersPaths[k].append([])  # Initialize list for this hop
            if hop == 0:
                # Add direct neighbors of pd
                for neighbor in transitions[pd]:
                    if neighbor not in visited:
                        evadersPaths[k][hop].append(bases.get(neighbor))
                        visited[neighbor] = hop
            else:
                prev_level_nodes = evadersPaths[k][hop - 1]
                for node_name in prev_level_nodes:
                    node_index = reverseBases.get(node_name)
                    for neighbor in transitions[node_index]:
                        if neighbor not in visited:
                            evadersPaths[k][hop].append(bases.get(neighbor))
                            visited[neighbor] = hop
    print()

def generateAllTransitions(gameGraph, pursuers, evaders):
    max = 0
    for pursuer in pursuers:
        if pursuer.getVelocity() > max:
            max = pursuer.getVelocity()

    for evader in evaders:
        if evader.getVelocity() > max:
            max = evader.getVelocity()

    global transitions
    transitions = []
    for k in range(len(gameGraph)):
        transitions.append([])
        visited = {}
        for hop in range(max):
            if hop == 0:
                transitions[k].append([])
                for j in range(len(gameGraph)):
                    if gameGraph[k][j] == 1 and visited.get(j) == None:
                        transitions[k][hop].append(j)
                        visited[j] = 0
            else:
                listVelocity = transitions[k][hop - 1]
                if hop < max: transitions[k].append([])
                for i in range(len(listVelocity)):
                    for j in range(len(gameGraph)):
                        if gameGraph[listVelocity[i]][j] == 1 and visited.get(j) == None:
                            transitions[k][hop].append(j)
                            visited[j] = 0

    print("Max Velocity: " + str(max))
    return transitions

def generateAllEvaderTransitions(transitions, vel):
    new_graph = [[] for _ in range(len(transitions))]

    for start in transitions:
        # Breadth-first search up to depth n
        visited = set()
        queue = deque([(start, 0)])
        visited.add(start)

        while queue:
            current, dist = queue.popleft()
            if 0 < dist <= vel:
                new_graph[start].append(current)

            if dist < vel:
                for neighbor in transitions[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

    return new_graph

def generateAllStates():
    global countCapture
    global statesFalse
    global statesTrue
    global total
    global activePDs
    global dirTransitions
    global dirStates
    global dirRemain
    global dirPerformed  # transtions ro and epsilon
    global enjoyStatesFalse
    global enjoyStatesTrue
    global linesStateFalse
    global linesStateTrue

    parallelTime = 0
    dirTransitions = output_dir+"/transitions/"
    dirStates = output_dir+"/states/"
    dirRemain = output_dir+"/remain/"
    dirPerformed = output_dir+"/performed/"

    for dir in[dirTransitions,dirStates,dirRemain,dirPerformed]:
        os.makedirs(dir, exist_ok=True)

    total = baseSPD * group - gap  # time and boolen of the enable or disable especial dot point
    common = 9999990

    statesFalse = []
    statesFalse.append([])
    statesFalse[0] = np.full((traditional), common, dtype=np.int32)
    statesTrue = []
    statesTrue.append([])
    statesTrue[0] = np.full((traditional), common, dtype=np.int32)
    print(str(sys.getsizeof(statesFalse[0])))
    print(str(sys.getsizeof(statesTrue[0])))

    enjoyStatesFalse = {}
    enjoyStatesTrue = {}
    linesStateFalse = {}
    linesStateTrue = {}

    allStates = 0
    countCapture = 0
    countSpd = 0

    print("Total Cores/Processes: " + str(multiprocessing.cpu_count()))
    print("Actives Cores: " + str(onCores))
    print("Agents: " + str(agents))
    print("-----------------------------------------------")
    print("Separating States")

    offset0 = calcOffSet(traditional, 0)
    block = (offset0 + multiprocessing.cpu_count()) // multiprocessing.cpu_count()

    totalFiles = (baseSPD - 1) * timeGroup + 1
    countFile = 0
    activePDs = []
    prefix = '0'

    print(str(sys.getsizeof(statesFalse[0])))
    print(str(sys.getsizeof(statesTrue[0])))
    for spdUnit in range(baseSPD):
        for t in range(timeGroup):
            prefix = convertToBin(spdUnit) + decToHex.get(t)
            # with open(dirStates + prefix + '-false.txt', 'w') as file:
            #     file.write("")
            # file.close()

            # with open(dirStates + prefix + '-true.txt', 'w') as file:
            #     file.write("")
            # file.close()

            if spdUnit > 0 and t < time:
                with open(dirStates + prefix + '-falseLine.txt', 'w') as file:
                    file.write("")
                file.close()

                with open(dirStates + prefix + '-trueLine.txt', 'w') as file:
                    file.write("")
                file.close()

            iterCount = 0
            totalPartial = traditional

            if spdUnit == 0 and time != 0 or spdUnit != 0 and t == time:  # if initial frame or last frame of a group
                pacDots = convertToBin(spdUnit)
                for pd in range(spd):
                    if pacDots[pd] == '0':
                        activePDs.append(specialPacDot[pd])
                capture = 0
                pacDot = 9999999
            elif spdUnit == 0 and time == 0 and timeGroup == 1:
                activePDs = []
                capture = 0
                pacDot = 9999990
            else:  # if intermediate frames (between first and penultimate) of a group
                activePDs = []
                capture = 9999999
                pacDot = 9999990

            countFile += 1
            print("Prefix Files: " + prefix + " - File " + str(countFile) + " of " + str(totalFiles))

            while iterCount < totalPartial:
                if totalPartial - iterCount >= offset0:
                    iterMax = offset0 + iterCount
                else:
                    iterMax = totalPartial
                    block = totalPartial % block
                print(
                    "InterCount: " + str(iterCount) + " - IterMax: " + str(iterMax) + " - Total: " + str(totalPartial))

                stParallel = timeit.default_timer()
                
                results = Parallel(n_jobs=cores, backend='multiprocessing')(
                    delayed(separateStates)(id, block, spdUnit, t, totalPartial)
                    for id in range(iterCount, iterMax, block)
                )

                for partial_result in results:
                    for state_type, state_id, feasible in partial_result:
                        if spdUnit > 0 and t < time:
                            if feasible:
                                if state_type == CAPTURE_STATE:
                                    enjoyStatesFalse[state_id] = capture
                                    enjoyStatesTrue[state_id] = capture
                                    countCapture += 1
                                elif state_type == PACDOT_STATE:
                                    enjoyStatesFalse[state_id] = pacDot
                                    enjoyStatesTrue[state_id] = pacDot
                                    countSpd += 1
                                elif state_type == EVADER_WIN_STATE:
                                    enjoyStatesFalse[state_id] = EVADER_WIN_VALUE
                                    enjoyStatesTrue[state_id] = EVADER_WIN_VALUE
                                else:
                                    enjoyStatesFalse[state_id] = common
                                    enjoyStatesTrue[state_id] = common
                                allStates += 1
                        else:
                            if state_type == CAPTURE_STATE:
                                statesFalse[0][state_id] = capture
                                statesTrue[0][state_id] = capture
                                countCapture += 1
                            elif state_type == PACDOT_STATE:
                                statesFalse[0][state_id] = pacDot
                                statesTrue[0][state_id] = pacDot
                                countSpd += 1
                            elif state_type == EVADER_WIN_STATE:
                                statesFalse[0][state_id] = EVADER_WIN_VALUE
                                statesTrue[0][state_id] = EVADER_WIN_VALUE
                            else:
                                statesFalse[0][state_id] = common
                                statesTrue[0][state_id] = common
                            allStates += 1

                fsParallel = timeit.default_timer()
                parallelTime += fsParallel-stParallel
                iterCount = iterMax
                block = (offset0 + multiprocessing.cpu_count()) // multiprocessing.cpu_count()

                # file writing here
                if spdUnit > 0 and t < time:
                    # with open(dirStates + convertToBin(spdUnit) + decToHex.get(t) + '-false.txt', "a") as file:
                    #     for falseId, falseCost in enjoyStatesFalse.items():
                    #         file.writelines(str(falseId) + '-' + str(falseCost) + '\n')
                    # file.close()

                    with open(dirStates + convertToBin(spdUnit) + decToHex.get(t) + '-falseLine.txt', "a") as file:
                        for falseId, falseCost in enjoyStatesFalse.items():
                            file.writelines(str(falseId) + '\n')
                    file.close()
                    enjoyStatesFalse.clear()

                    # with open(dirStates + convertToBin(spdUnit) + decToHex.get(t) + '-true.txt', "a") as file:
                    #     for trueId, trueCost in enjoyStatesTrue.items():
                    #         file.writelines(str(trueId) + '-' + str(trueCost) + '\n')
                    # file.close()

                    with open(dirStates + convertToBin(spdUnit) + decToHex.get(t) + '-trueLine.txt', "a") as file:
                        for trueId, trueCost in enjoyStatesTrue.items():
                            file.writelines(str(trueId) + '\n')
                    file.close()
                    enjoyStatesTrue.clear()
                # else:
                #     with open(dirStates + convertToBin(spdUnit) + decToHex.get(t) + '-false.txt', "a") as file:
                #         for false in statesFalse[0]:
                #             file.writelines(str(false) + '\n')
                #     file.close()

                #     with open(dirStates + convertToBin(spdUnit) + decToHex.get(t) + '-true.txt', "a") as file:
                #         for true in statesTrue[0]:
                #             file.writelines(str(true) + '\n')
                #     file.close()

            if spdUnit == 0 and t == 0:
                break

        if timeGroup == 1:
            break

    print("Enjoy: "+str(sys.getsizeof(enjoyStatesTrue)))
    print("Enjoy: "+str(sys.getsizeof(enjoyStatesFalse)))

    print("\nPartial 1 - Parallel Time: " + str(int(parallelTime / 60)) + ":" + str(math.ceil(parallelTime % 60)) + '\n')

    print("\n-----------------------------------------------")
    print("Generating Transitions")  # , backend='multiprocessing',

    iterCount = 0
    iterMax = 0
    totalPartial = traditional
    gc.collect()

def generateAllSpecialTransitions():
    global specialTransitions
    global specialPacDotFactor
    specialTransitions = []
    specialPacDotFactor = []

    identity = []
    for i in range(spd):  # spd = number of special pac-dots
        identity.append([])
        for j in range(spd):
            identity[i].append('1' if i == j else '0')

    for k in range(baseSPD):  # binary base with spd power
        specialTransitions.append([])
        specialPacDotFactor.append([])
        tuple = convertToBin(k)
        for i in range(spd):
            out = ''
            for j in range(spd):
                if tuple[j] != identity[i][j]:
                    out += '1'
                else:
                    out += '0'

            if out > tuple:
                specialTransitions[k].append(out)
            else:
                specialTransitions[k].append('0')
                specialPacDotFactor[k].append(i)
    print()

CAPTURE_STATE=0
PACDOT_STATE=1
REGULAR_STATE=2
EVADER_WIN_STATE=3

EVADER_WIN_VALUE=1000000

def separateStates(idBase, block, spdUnit, t, totalParcial):
    returned = []
    bound = totalParcial if idBase + block > totalParcial else idBase + block

    for id in range(idBase, bound):
        tuple = convertToTupleBase(id)
        feasibleState = False
        if spdUnit > 0 and t < time:
            feasibleState = isFeasibleState(tuple, spdUnit, t)
        
        remainder=id
        for i in range(len(evaders)):
            evader_pos=remainder%base
            remainder//=base
            if evader_pos == base-1: # Terminal state
                returned.append((EVADER_WIN_STATE, id, feasibleState))
                break
        else:
            pacDot = False
            for pd in activePDs:
                if tuple[len(pursuers) * lengthSimbol:] == pd:
                    pacDot = True
                    break
            if pacDot:  # if special Pac-Dot, it has bigger precedence then capture
                returned.append((1, id, feasibleState))
            elif isCaptureState(tuple, len(pursuers)):  # if is a capture state
                returned.append((0, id, feasibleState))
            else:  # if is common state
                returned.append((2, id, feasibleState))

    return returned


def isFeasibleCompleteState(state, spdUnit, t):
    if spdUnit == 0 or spdUnit > 0 and t == time:
        return True
    else:
        return isFeasibleState(state[overHead:], spdUnit, t)


def isFeasibleState(tuple, spdUnit, t):
    if spdUnit == 0 or spdUnit > 0 and t == time:
        return True
    else:
        for k in specialPacDotFactor[spdUnit]:
            for x in range(t + 1):
                for y in range(len(evadersPaths[k][x])):
                    if tuple[len(pursuers) * lengthSimbol:] == evadersPaths[k][x][y]:
                        return True
        return False


def selectMakedTurnFalseStates(dict_, index, spdUnit, t, idOffset):
    keys = list(dict_.keys())
    num_keys = len(keys)
    results = np.full((num_keys, 3), 9999990, dtype=np.int32)

    index2 = 0

    for temp in dict_.keys():
        line = generateStateTransitionsTrue(temp, idOffset, spdUnit, t)
        dict_[temp].extend(line)

    for idx, temp in enumerate(keys):
        adjacents = dict_[temp]
        stateTrue = statesTrue[index][temp]

        if stateTrue >= 9999990 and stateTrue != 100000001 and len(adjacents) > 0:
            adj_arr = np.array(adjacents, dtype=np.int32)
            state_false_values = statesFalse[index2][adj_arr]

            if not np.any((state_false_values == 9999990) | (state_false_values == 9999999)):
                max_idx = np.argmax(state_false_values)
                new_value = state_false_values[max_idx]
                max_state = adj_arr[max_idx]
                results[idx] = (new_value, temp, max_state)

    return results


def selectMakedTurnTrueStates(dict_, index, spdUnit, t, idOffset):
    keys = list(dict_.keys())
    num_keys = len(keys)
    results = np.full((num_keys, 3), 9999990, dtype=np.int32)

    for idx, temp in enumerate(keys):
        line = generateStateTransitionsFalse(temp, idOffset, spdUnit, t)
        dict_[temp].extend(line)
    
    for idx, temp in enumerate(keys):
        adjacents = dict_[temp]
        stateFalse = statesFalse[index][temp]

        if stateFalse in (9999990, 9999999) and len(adjacents) > 0:
            adj_array = np.array(adjacents, dtype=np.int32)
            true_values = statesTrue[index][adj_array]

            min_idx = np.argmin(true_values)
            min_val = true_values[min_idx]

            if min_val<9999990:
                results[idx] = (min_val + (-1 if min_val>EVADER_WIN_VALUE/2 else 1), temp, adj_array[min_idx])

    return results


def calcOffSet(totalParcial, factor):
    if totalParcial <= 1000000:
        return totalParcial
    else:
        factor2 = (totalParcial // 1000000) + factor
        return (totalParcial + factor2) // factor2


def processGame():
    global ro
    global epsilon
    global UFalse
    global UTrue
    global statesTrue
    global statesFalse

    ro = []
    epsilon = []
    UFalse = []
    UTrue = []
    change = True
    global countTrue
    global countFalse
    countTrue = 0
    countFalse = 0
    offset2 = calcOffSet(traditional, 4)
    block = (offset2 + multiprocessing.cpu_count()) // multiprocessing.cpu_count()
    controlle1 = 0
    controlle2 = 0
    previous1 = 0
    previous2 = 0
    idOffset = 0
    parallelTime = 0

    i = 0
    agentsStr = str(agents)

    global fileRo
    global fileEpsilon
    global fileRoLine
    global fileEpsilonLine

    fileRo = tagName + '-' + agentsStr + '-' + speedEachOne + '-ro.txt'
    fileEpsilon = tagName + '-' + agentsStr + '-' + speedEachOne + '-epsilon.txt'
    fileRoLine = tagName + '-' + agentsStr + '-' + speedEachOne + '-roLine.txt'
    fileEpsilonLine = tagName + '-' + agentsStr + '-' + speedEachOne + '-epsilonLine.txt'

    partial_state_path = os.path.join(dirPerformed, f"{tagName}-{agentsStr}-{speedEachOne}-partial.pkl")
    # if os.path.exists(partial_state_path):
    #     print("Loading partial state from disk...")
    #     partial = load_partial_state(partial_state_path)
    #     statesFalse = partial['statesFalse']
    #     statesTrue = partial['statesTrue']
    #     countTrue = partial['countTrue']
    #     countFalse = partial['countFalse']
    #     controlle1 = partial['controlle1']
    #     controlle2 = partial['controlle2']
    #     i = partial['i']
    #     roLine = partial['roLine']
    #     epsilonLine = partial['epsilonLine']
    # else:
    #     print("No partial state found, starting from scratch...")

    # Files ro and epsilon to save performed transtitions
    with open(dirPerformed + fileRo, 'w') as file:
        file.write('')
        file.close()

    with open(dirPerformed + fileEpsilon, 'w') as file:
        file.write('')
        file.close()

    with open(dirPerformed + fileRoLine, 'w') as file:
        file.write('')
        file.close()

    with open(dirPerformed + fileEpsilonLine, 'w') as file:
        file.write('')
        file.close()

    hasFileRemainFalse = []
    hasFileRemainTrue = []

    for it in range(baseSPD):
        hasFileRemainFalse.append([])
        hasFileRemainTrue.append([])
        for it2 in range(timeGroup):
            hasFileRemainFalse[it].append(False)
            hasFileRemainTrue[it].append(False)
        if timeGroup == 1:
            break

    print("-----------------------------------------------")
    print("Process States Costs")
    print('\n=====================')

    prefix = ''

    roLine = 0
    epsilonLine = 0
    while change:
        change = False
        index = 0
        index2 = 0
        idOffset = 0
        shift1 = baseSPD if timeGroup == 1 else 1
        shift2 = 0 if timeGroup == 1 else 1

        for spdUnit in range(baseSPD - shift1, -1, -1):
            if spdUnit == 0:
                shift2 = timeGroup
            for t in range(timeGroup - shift2, -1, -1):
                prefix = convertToBin(spdUnit) + decToHex.get(t)

                if spdUnit > 0:
                    idOffset = spdUnit * group + t * traditional - gap
                else:
                    idOffset = 0

                iterCount = 0
                iterMax = 0
                invite = []
                totalFalse = len(statesFalse[index])
                while iterCount < totalFalse:
                    if totalFalse - 1 - iterCount > offset2:
                        iterMax = offset2 + iterCount
                        block = (offset2 + multiprocessing.cpu_count()) // multiprocessing.cpu_count()
                    else:
                        iterMax = totalFalse
                        block = (iterMax - iterCount + multiprocessing.cpu_count()) // multiprocessing.cpu_count()

                    for k in range(iterCount, iterMax):
                        stateValue = statesFalse[index][k]

                        if stateValue == 9999990 or stateValue == 9999999:
                            controlle1 += 1
                            if len(invite) == 0 or len(invite[len(invite) - 1]) % block == 0:
                                invite.append({})
                            invite[len(invite) - 1][k] = []

                    if len(invite) > 0:
                        stParallel = timeit.default_timer()
                        result = Parallel(n_jobs=cores, backend='multiprocessing', ) \
                            (delayed(selectMakedTurnTrueStates)(invite[temp], index, spdUnit, t, idOffset) for temp in range(len(invite)))
                        invite.clear()
                        fsParallel = timeit.default_timer()
                        parallelTime += fsParallel-stParallel

                        valid_mask = [(result1[:,0] < 9999990) | (result1[:,0] > 9999999) for result1 in result]

                        for result1,valid in zip(result,valid_mask):
                            if np.any(valid):
                                change = True
                                break

                        if change:
                            for result1,valid in zip(result,valid_mask):
                                for r in result1[valid]:
                                    statesFalse[index2][r[1]] = r[0]
                                    previous1 += 1
                                    countFalse += 1
                                    ro.append(Transition(r[1] + idOffset, r[2], r[0]))

                        # save ro performed transitions in file
                        with open(dirPerformed + fileRo, 'a') as file:
                            for it in ro:
                                file.write(
                                    str(it.getOrigin()) + ' ' + str(it.getDestiny()) + ' ' + str(it.getCost()) + '\n')
                        file.close()
                        if len(ro) > 0:
                            with open(dirPerformed + fileRoLine, 'a') as file:
                                file.write(str(roLine + 1) + ' ' + str(roLine + len(ro)) + '\n')
                                roLine += len(ro)
                        ro.clear()

                    iterCount = iterMax
                    print(datetime.now(),iterCount)
                    gc.collect()

                print(prefix + " - " + str(i + 1) + " C1 " + str(controlle1 - previous1))

                controlle1 = 0
                previous1 = 0

                print('---------------------')

                iterCount = 0
                iterMax = 0
                invite = []

                totalTrue = len(statesTrue[index])
                while iterCount < totalTrue:
                    if totalTrue - 1 - iterCount > offset2:
                        iterMax = offset2 + iterCount
                        block = (iterMax + multiprocessing.cpu_count()) // multiprocessing.cpu_count()
                    else:
                        iterMax = totalTrue
                        block = (iterMax - iterCount + multiprocessing.cpu_count()) // multiprocessing.cpu_count()

                    for k in range(iterCount, iterMax):
                        stateValue = statesTrue[index][k]

                        if stateValue == 9999990 or stateValue == 9999999:
                            controlle2 += 1
                            if len(invite) == 0 or len(invite[len(invite) - 1]) % block == 0:
                                invite.append({})
                            invite[len(invite) - 1][k] = []

                    if len(invite) > 0:
                        stParallel = timeit.default_timer()
                        result = Parallel(n_jobs=cores, backend='multiprocessing', ) \
                            (delayed(selectMakedTurnFalseStates)(invite[temp], index, spdUnit, t, idOffset) for temp in range(len(invite)))
                        invite.clear()
                        fsParallel = timeit.default_timer()
                        parallelTime += fsParallel-stParallel

                        valid_mask = [(result1[:,0] < 9999990) | (result1[:,0] > 9999999) for result1 in result]

                        if not change:
                            for result1,valid in zip(result,valid_mask):
                                if np.any(valid):
                                    change = True
                                    break

                        if change:
                            for result1,valid in zip(result,valid_mask):
                                for r in result1[valid]:
                                    statesTrue[index2][r[1]] = r[0]
                                    previous1 += 1
                                    countTrue += 1
                                    epsilon.append(Transition(r[1] + idOffset, r[2], r[0]))

                        # save epsilon performed transitions in file
                        with open(dirPerformed + fileEpsilon, 'a') as file:
                            for it in epsilon:
                                file.write(
                                    str(it.getOrigin()) + ' ' + str(it.getDestiny()) + ' ' + str(it.getCost()) + '\n')
                        file.close()
                        if len(epsilon) > 0:
                            with open(dirPerformed + fileEpsilonLine, 'a') as file:
                                file.write(str(epsilonLine + 1) + ' ' + str(epsilonLine + len(epsilon)) + '\n')
                                epsilonLine += len(epsilon)
                        epsilon.clear()
                
                    iterCount = iterMax
                    print(datetime.now(),iterCount)
                    gc.collect()
                    file.close()

                save_partial_state(partial_state_path, {
                    'statesFalse': statesFalse,
                    'statesTrue': statesTrue,
                    'countTrue': countTrue,
                    'countFalse': countFalse,
                    'controlle1': controlle1,
                    'controlle2': controlle2,
                    'i': i,
                    'roLine': roLine,
                    'epsilonLine': epsilonLine,
                })
                print(f"Progress saved at iteration {i}.")

                print(prefix + " - " + str(i + 1) + " C2 " + str(controlle2 - previous2))

                controlle2 = 0
                previous2 = 0

                print('=====================')
        i = i + 1

    print("\nCountTrue: " + str(countTrue) + "\nCountFalse: " + str(countFalse) + "\nCountCaputure: " + str(
        countCapture) + "\nCountCaputure: " + str(countCapture))
    
    linesStateTrue.clear()
    linesStateFalse.clear()
    enjoyStatesTrue.clear()
    enjoyStatesFalse.clear()
    gc.collect()

    # if os.path.exists(partial_state_path):
    #     os.remove(partial_state_path)
    #     print("Removed partial state file.")

    print("\nPartial 2 - Parallel Time: " + str(int(parallelTime / 60)) + ":" + str(math.ceil(parallelTime % 60)) + '\n')

    return countTrue + countFalse + countCapture + countCapture, index2

def preProcessGraph(gameGraph):
    global weights
    weights = []

    for i in range(len(gameGraph)):
        weights.append([])
        for j in range(len(gameGraph)):
            if i != j and gameGraph[i][j] == 1:
                weights[i].append(1)
            else:
                weights[i].append(0)

    removed = []

    for i in range(len(gameGraph)):
        adj1 = adj2 = 0
        count = 0
        for j in range(len(gameGraph)):
            if i != j and gameGraph[i][j] == 1:
                count += 1
                if count == 1:
                    adj1 = j
                elif count == 2:
                    adj2 = j

        if count == 2:
            if gameGraph[adj1][adj2] == 0 or gameGraph[adj2][adj1] == 0:
                gameGraph[adj1][adj2] = 1
                gameGraph[adj2][adj1] = 1
                gameGraph[adj1][i] = 0
                gameGraph[adj2][i] = 0
                weights[adj1][adj2] += weights[adj1][i] + weights[adj2][i]
                weights[adj2][adj1] += weights[adj2][i] + weights[adj1][i]
                weights[adj1][i] = 0
                weights[adj2][i] = 0
                removed.append(i)

    for i in range(len(removed) - 1, -1, -1):
        for j in range(len(gameGraph)):
            del (gameGraph[j][removed[i]])
            del (weights[j][removed[i]])

    for i in range(len(removed) - 1, -1, -1):
        del (gameGraph[removed[i]])
        del (weights[removed[i]])

    print(gameGraph)
    print(weights)
    print("fim")


def get_hop_adjacency(graph, hops):
    """
    Computes nodes reachable in exactly 1 to `hops` steps from each node.

    Parameters:
    - graph: list[list[int]] - adjacency list (graph[i] are neighbors of node i)
    - hops: int - number of hops to consider

    Returns:
    - list[list[list[int]]] - result[n][i] contains list of nodes reachable from node n in exactly i+1 hops
    """
    num_nodes = len(graph)
    result = [[[] for _ in range(hops)] for _ in range(num_nodes)]

    for start in range(num_nodes):
        visited = {start: 0}
        queue = deque([(start, 0)])

        while queue:
            node, depth = queue.popleft()

            if depth == hops:
                continue

            for neighbor in graph[node]:
                if neighbor not in visited or visited[neighbor] > depth + 1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))
                    if neighbor not in result[start][depth]:
                        result[start][depth].append(neighbor)

    return result

def visualize_map(nodes, escape_positions, center, avg_coords, cos_lat):
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
    
    html_template = f"""
    <!DOCTYPE html>
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

                <span id="state_cost" style="margin-left:auto; font-weight:bold;">Cost: —</span>
            </div>
        </div>
        <div id="map"></div>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            const map = L.map('map').setView([{center[0]}, {center[1]}], {zoom_level});

            L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 19,
                attribution: '© OpenStreetMap'
            }}).addTo(map);
            
            const degToRad=Math.PI/180;
            function equirectangularProj(lat,lon){{
                return[
                    {earth_radius_m}*(lon{(-avg_coords[1]):+})*degToRad*{cos_lat},
                    {earth_radius_m}*(lat{(-avg_coords[0]):+})*degToRad
                ];
            }}

            const nodes = {json.dumps(map_nodes)};
            const transitions = {json.dumps([[x for xs in outgoing for x in xs] for outgoing in transitions])}; // Transitions as list of lists
            const counts = [{len(pursuers)}, {len(evaders)}];
            const db="{tagName}";
        </script>
        <script src="../static/js.js"></script>
    </body>
    </html>
    """

    with open("templates/"+tagName+".html", "w") as map_file:
        map_file.write(html_template)

def store_best_moves():
    conn = sqlite3.connect(output_dir+"/"+tagName+".db")
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS best_moves")
    cur.execute("CREATE TABLE best_moves (turn INTEGER, state_id INTEGER, best_move INTEGER, cost INTEGER, PRIMARY KEY(turn,state_id))")

    q="INSERT INTO best_moves VALUES (?, ?, ?, ?)"
    for turn,path in enumerate([fileRo,fileEpsilon]):
        with open(dirPerformed+path, "r") as f:
            batch = []
            for line in tqdm(f):
                origin, dest, cost = map(int, line.strip().split())
                batch.append((turn, origin, dest, cost))
                if len(batch) >= 10000:
                    cur.executemany(q, batch)
                    batch.clear()

            if batch:
                cur.executemany(q, batch)
    
    conn.commit()
    conn.close()

# @profile
def main(args):
    global globalTime
    global pursuers
    global evaders
    global time
    global reborn
    global tagName
    global parallelTime
    global output_dir
    reborn = []
    globalTime = timeit.default_timer()

    """tagName = 'Grid4X4'
    print("Grid 4X4")
    #			  0 1 2 3 4 5 6 7 8 9101112131415
    gameGraph = [[1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
                [0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0],
                [0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0],
                [1,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0],
                [0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0],
                [0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0],
                [0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0],
                [0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0],
                [0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,1],
                [0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1]]
    pursuers = [Pursuer(3, 1), Pursuer(3, 1), Pursuer(3, 1), Pursuer(3, 1)]
    evaders = [Evader(0, 1)]
    pacDots = [4]#[4, 12, 15]
    time = 0
    reborn.append(['15', '2'])"""

    global specialPacDot
    global overHead
    global traditional
    global timeGroup
    global group
    global spd
    global baseSPD
    global baseBin
    global gap
    global caracters
    global logFile
    global cores
    global onCores
    global speedEachOne
    global base
    
    if args.tag==None:
        chars = string.ascii_letters + string.digits  # a-zA-Z0-9
        tagName=''.join(random.choices(chars, k=10))
    else:
        if re.search(r"[^A-Za-z0-9_]",args.tag):
            raise ValueError("Tag must be a valid identifier.")
        tagName=args.tag
    
    output_dir=args.output_dir

    nodes,graph,escape_positions,avg_coords,cos_lat=make_map(args.osm_id,args.center,args.radius,args.granularity)
    print(len(graph),"nodes")

    # Add terminal state
    for node in range(len(graph)):
        graph[node].append(len(graph))
    graph.append([])

    # TODO: Support speed and PacDots
    pursuers=[Pursuer(1) for _ in range(args.pursuers)]
    evaders=[Evader(1) for _ in range(args.evaders)]
    pacDots=[]
    time=0
    reborn=[]

    global transitions
    transitions=get_hop_adjacency(graph,max([agent.velocity for agent in pursuers+evaders]))
    base=len(transitions)

    speedEachOne = ''
    for i in range(len(pursuers)):
        speedEachOne += str(pursuers[i].getVelocity())

    speedEachOne += str(evaders[0].getVelocity())

    parallelTime = 0

    cores = args.cores
    onCores = multiprocessing.cpu_count() + cores + 1 if cores<0 else cores

    logFile = str(len(pursuers)+len(evaders)) + '-' + str(onCores) + '-' + tagName + '-' + speedEachOne + '-' + datetime.now().strftime('%d-%m-%Y-%H-%M-%S') + '.txt'

    baseBin = 2
    spd = len(pacDots)
    baseSPD = 2 ** spd if time > 0 else 1
    overHead = spd + 1

    inicio = timeit.default_timer()
    generateBases(pursuers, evaders)
    generatePowers(pursuers, evaders)  # Validated
    fim = timeit.default_timer()
    totalTime = int(fim - inicio)
    print("Time Powers: " + str(int(totalTime / 60)) + ":" + str(totalTime % 60))

    traditional = base ** agents  # frame
    timeGroup = time + 1
    group = timeGroup * traditional
    gap = group - traditional

    specialPacDot = []
    for pac in pacDots:
        pd = convertToTupleBase(pac)
        specialPacDot.append(pd[len(pd) - lengthSimbol:])  # [trash:]) # Special Pac-Dots position list

    pathsFromPacDots(transitions,specialPacDot)

    generateAllSpecialTransitions()

    caracters = 'X'
    for i in range(1, lengthSimbol, 1):
        caracters += 'X'

    inicio = timeit.default_timer()
    generateAllStates()  # Validated
    fim = timeit.default_timer()
    totalTime = int(fim - inicio)
    print("Time AllSates: " + str(int(totalTime / 60)) + ":" + str(totalTime % 60))

    inicio = timeit.default_timer()
    tuple = processGame()

    fim = timeit.default_timer()
    totalTime = int(fim - inicio)
    print("Time Process: " + str(int(totalTime / 60)) + ":" + str(totalTime % 60))
    print("States: " + str(len(statesFalse[0]) + len(statesTrue[0])))

    store_best_moves()
    visualize_map(nodes, escape_positions, args.center, avg_coords, cos_lat)

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

map.doubleClickZoom.disable();

let turn = 0;
const PURSUER = 0,
  EVADER = 1;
let movingAgent = null;
let history = [];
let highlightedNodes = [];
let agentList = [];
let moved = [];
let apiData, apiProm;
let stateID = 0;
let turnStateID;
let gameEnded=false;
let escaped=[];
const numStates = nodes.length+1; // Nodes + terminal state

const topBarProper= document.getElementById("top_bar_proper");
const placeAgents = document.getElementById("place_agents");

class Agent {
  constructor(marker, index, type, node, factor) {
    this.marker = marker; // DOM Node
    this.index = index; // [0,counts[type])
    this.type = type; // {PURSUER, EVADER}
    this.node = node; // from nodes list
    this.factor = factor; // numStates**k, k in [0,numAgents)
  }
}

class MoveHistory {
  constructor(agent, from) {
    this.agent = agent; // Agent object
    this.from = from; // from nodes list
  }
}

class TurnHistory {
  constructor(moved,turnStateID) {
    this.moved = moved; // who moved last turn
    this.turnStateID = turnStateID;
  }
}

const iconSrcs=["../static/ghost.png", "../static/pacman.png"];

function createMarkerFromNode(node, isPursuer) {
  const src = iconSrcs[1-isPursuer];
  const size = isPursuer ? [32, 32] : [28, 30];
  const icon = L.icon({ iconUrl: src, iconSize: size });
  return L.marker([node.lat, node.lon], { icon }).addTo(map);
}

function createAgentFromNode(node) {
  return {
    lat: node.lat,
    lon: node.lon,
    pos: node.id,
  };
}

function parseAgentCoords() {
  const params = new URLSearchParams(window.location.search);
  const parseList = (key) => {
    const val = params.get(key);
    if (!val) return [];
    return val.split(";").map((s) => {
      const [lat, lon] = s.split(",").map(Number);
      return { lat, lon };
    });
  };
  let evaders = parseList("ep"); // ep = evader positions
  if (evaders.length != counts[EVADER]) {
    return [];
  }
  let pursuers = parseList("pp"); // pp = pursuer positions
  if (pursuers.length != counts[PURSUER]) {
    return [];
  }
  return [...pursuers, ...evaders];
}

function updateURLWithAgentCoords() {
  const pursuersCoords = agents
    .slice(0, counts[0])
    .map((a) => `${a.lat},${a.lon}`)
    .join(";");
  const evadersCoords = agents
    .slice(counts[0])
    .map((a) => `${a.lat},${a.lon}`)
    .join(";");

  const url = new URL(window.location.href);
  url.searchParams.set("pp", pursuersCoords);
  url.searchParams.set("ep", evadersCoords);
  window.history.replaceState(null, "", url.toString());
}

function setupAgentsAndTransitions() {
  topBarProper.classList.remove("display_none");

  for (let i = 0; i < transitions.length; i++) {
    transitions[i] = new Set(transitions[i]);
  }

  let factor = numStates ** (counts[0] + counts[1]);
  agents.forEach((a, i) => {
    if (!a.marker) {
      a.marker = createMarkerFromNode(nodes[a.pos], i < counts[PURSUER]);
    }

    factor /= numStates;
    let agent = new Agent(
      a.marker,
      i,
      i < counts[PURSUER] ? PURSUER : EVADER,
      nodes[a.pos],
      factor
    );
    stateID += factor * a.pos;
    agentList.push(agent);
    a.marker.on("click", (e) => {
      agentClick(agent);
    });
  });
  turnStateID = stateID;

  nodes.forEach((n) => {
    n.marker.on("click", function () {
      nodeClick(n);
    });
  });

  updateCostAndBestMove();
  for (let a of agentList) {
    if (a.type != turn) {
      a.marker._icon.classList.add("hoverable");
    }
  }

  // Load local storage
  ["ai_pursuer", "ai_evader"].forEach((id) => {
    const saved = localStorage.getItem(id);
    if (saved !== null) {
      document.getElementById(id).checked = saved == "true";
    }
  });

  const speed = localStorage.getItem("ai_speed");
  if (speed !== null) {
    document.getElementById("ai_speed").value = speed;
  }

  if (
    document.getElementById("ai_pursuer").checked ||
    document.getElementById("ai_evader").checked
  ) {
    startAIPlay();
  }
}

let hoverPreview = null;
function promptUserForAgentPositions() {
  placeAgents.classList.remove("display_none");
  const agentsDiv=document.getElementById("agents");
  for(let i=0; i<counts[PURSUER]; i++){
    const img=document.createElement("img");
    img.src=iconSrcs[0];
    agentsDiv.appendChild(img);
  }
  for (let i = 0; i < counts[EVADER]; i++) {
    const img = document.createElement("img");
    img.src = iconSrcs[1];
    agentsDiv.appendChild(img);
  }

  let needed = counts[0] + counts[1];

  nodes.forEach((n) => {
    n.marker.on("click", function handler(e) {
      const agent = createAgentFromNode(n);
      agent.marker = hoverPreview;
      hoverPreview.getElement()?.classList.remove("transp");
      agents.push(agent);
      hoverPreview = null;
      agentsDiv.removeChild(agentsDiv.firstElementChild);

      if (agents.length == needed) {
        // Clean up listeners
        nodes.forEach((n2) => {
          n2.marker.off("click", handler);
        });
        nodes.forEach((n2) => {
          n2.marker.off("mouseover");
        });
        nodes.forEach((n2) => {
          n2.marker.off("mouseout");
        });
        updateURLWithAgentCoords();
        placeAgents.classList.add("display_none");
        setupAgentsAndTransitions();
      }
    });

    n.marker.on("mouseover", () => {
      if (agents.length >= needed) return;
      if (hoverPreview) {
        map.removeLayer(hoverPreview);
      }
      const isPursuer = agents.length < counts[0];
      hoverPreview = createMarkerFromNode(n, isPursuer);
      L.setOptions(hoverPreview, { interactive: false });
      hoverPreview.getElement()?.classList.add("transp");
    });

    n.marker.on("mouseout", () => {
      if (hoverPreview) {
        map.removeLayer(hoverPreview);
        hoverPreview = null;
      }
    });
  });
}

function passTurnBtn(){
  stopAIPlay();
  passTurn();
  startAIPlay();
}

function passTurn() {
  history.push(new TurnHistory(moved, turnStateID));
  moved = [];
  turn = 1 - turn;
  turnStateID=stateID;
  updateCostAndBestMove();
  for (let a of agentList) {
    a.marker._icon.classList.toggle("hoverable");
  }
}

async function makeBestMove() {
  if(gameEnded) return false;
  await apiProm;

  if(apiData.cost==1000000){
    gameEnded = true;
    stopAIPlay();

    let remainder = apiData.move % numStates**counts[EVADER]; 
    for (let i=counts[EVADER]; i<agentList.length; i++) {
      const a=agentList[i];
      const newNodeID = Math.floor(remainder / a.factor);
      remainder -= newNodeID * a.factor;
      if (newNodeID == nodes.length) {
        escaped.push(a.marker);
        a.marker.getElement()?.classList.add("display_none");
      }
    }
    
    return;
  }

  let remainder = apiData.move;

  for (const a of agentList) {
    const newNodeID = Math.floor(remainder / a.factor);
    remainder -= newNodeID * a.factor;
    if (newNodeID != a.node.id) {
      const newNode = nodes[newNodeID];
      if (!newNode) {
        return false;
      }

      history.push(new MoveHistory(a, a.node));
      moveAgent(a, a.node, newNode);
      moved.push(a);
    }
  }
  moved.pop();
  passTurn();
  return true;
}

function moveAgent(agent, oldNode, newNode) {
  stateID += (newNode.id - oldNode.id) * agent.factor;
  agent.marker.setLatLng([newNode.lat, newNode.lon]);
  agent.node = newNode;
}

function undo() {
  for(let v of escaped){
    v.getElement()?.classList.remove("display_none");
  }
  escaped=[];

  if (history.length == 0) {
    return false;
  }
  stopAIPlay();
  startAIPlay();

  let lastTurn=null;
  let last = history.pop();
  while (last instanceof TurnHistory) {
    turn = 1 - turn;
    lastTurn = last;
    last = history.pop();
  }

  if (lastTurn) {
    moved = lastTurn.moved;
    turnStateID = lastTurn.turnStateID;
    updateCostAndBestMove();
  } else {
    moved.pop();
  }

  moveAgent(last.agent, last.agent.node, last.from);
  movingAgent = null;
  clearHighlights();
  gameEnded=false;
  return true;
}

function reset(){
  const url = location.href;
  location.href = url.substring(0, url.indexOf("?"));
}

function nodeClick(n){
  if (movingAgent == null || !transitions[movingAgent.node.id].has(n.id)) {
    return;
  }

  history.push(new MoveHistory(movingAgent, movingAgent.node));
  moveAgent(movingAgent, movingAgent.node, n);

  if (moved.length == counts[turn] - 1) {
    passTurn();
  } else {
    moved.push(movingAgent);
  }

  movingAgent.marker._icon.classList.remove("selected-agent");
  clearHighlights();
  movingAgent = null;
}

function agentClick(a) {
  if (turn != a.type || moved.includes(a)) {
    nodeClick(a.node);
    return;
  }
  if (movingAgent == a) {
    movingAgent.marker._icon.classList.remove("selected-agent");
    movingAgent = null;
    clearHighlights();
    return;
  }

  if (movingAgent != null) {
    movingAgent.marker._icon.classList.remove("selected-agent");
  }

  movingAgent = a;
  movingAgent.marker._icon.classList.add("selected-agent");
  highlightTransitions(movingAgent.node.id);
}

function highlightTransitions(fromID) {
  clearHighlights();
  transitions[fromID].forEach((id) => {
    const node = nodes[id];
    if (node && node.marker) {
      node.marker._path.classList.add("highlighted-transition");
      highlightedNodes.push(node.marker);
    }
  });
}

function clearHighlights() {
  highlightedNodes.forEach((m) =>
    m._path.classList.remove("highlighted-transition")
  );
  highlightedNodes = [];
}

const costElem = document.getElementById("state_cost");
function updateCostAndBestMove() {
  if(gameEnded) return;
  apiProm = fetch(`/best_move?state_id=${turnStateID}&turn=${turn}&db=${db}`);

  apiProm.then(async function (res) {
    if (!res.ok){
      costElem.innerText = `Cost: 0`;
      gameEnded = true;
      stopAIPlay();
      return null;
    }

    apiData = await res.json();
    // if (apiData.cost == 1000000) {
    //   gameEnded = true;
    //   stopAIPlay();
    // }

    if (!apiData || apiData.cost == null) {
      costElem.innerText = "Cost: —";
    } else {
      costElem.innerText = `Cost: ${apiData.cost}`;
    }
  });
}

// AI play

let aiInterval = null;

function startAIPlay() {
  if (aiInterval) return;
  const intv = 50000 / document.getElementById("ai_speed").value;

  aiInterval = setInterval(() => {
    let pursuerChecked = document.getElementById("ai_pursuer").checked;
    let evaderChecked = document.getElementById("ai_evader").checked;

    if (
      (turn === PURSUER && pursuerChecked) ||
      (turn === EVADER && evaderChecked)
    ) {
      if (!makeBestMove()) {
        clearInterval(aiInterval);
        aiInterval = null;
      }
    }
  }, intv);
}

function stopAIPlay() {
  if (aiInterval) {
    clearInterval(aiInterval);
    aiInterval = null;
  }
}

function findNearestNode(lat, lon) {
  const [x, y] = equirectangularProj(lat, lon);
  let minDist = Infinity;
  let closestId = null;

  for (const node of nodes) {
    const [nx, ny] = node.proj;
    const dx = nx - x;
    const dy = ny - y;
    const distSq = dx * dx + dy * dy;

    if (distSq < minDist) {
      minDist = distSq;
      closestId = node.id;
    }
  }

  return closestId;
}

["ai_pursuer", "ai_evader", "ai_speed"].forEach((id) => {
  document.getElementById(id).addEventListener("change", () => {
    const el = document.getElementById(id);
    if (el.type == "checkbox") {
      localStorage.setItem(id, el.checked);
    } else {
      localStorage.setItem(id, el.value);
    }

    stopAIPlay();
    if (
      document.getElementById("ai_pursuer").checked ||
      document.getElementById("ai_evader").checked
    ) {
      startAIPlay();
    }
  });
});

const coords = parseAgentCoords();
const agents = [];

nodes.forEach((n) => {
  const marker = L.circleMarker([n.lat, n.lon], {
    radius: 7,
    color: n.color,
    fillColor: n.color,
    fill: true,
    fillOpacity: 0.3,
  });
  marker.addTo(map);
  n.marker = marker;
});

// Start
if (coords.length == 0) {
  promptUserForAgentPositions();
} else {
  let i = 0;
  for (const { lat, lon } of coords) {
    const nearest = findNearestNode(lat, lon);
    if (nearest != null) {
      let a = createAgentFromNode(nodes[nearest]);
      a.marker = createMarkerFromNode(nodes[nearest], i < counts[PURSUER]);
      agents.push(a);
    }
    i++;
  }

  setupAgentsAndTransitions();
}

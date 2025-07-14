map.doubleClickZoom.disable();
nodes.forEach((n,i) => {
  const marker = L.circleMarker([n.lat, n.lon], {
    radius: 7,
    color: n.color,
    fillColor: n.color,
    fill: true,
    fillOpacity: 0.3,
  });
  marker.addTo(map);
  n.marker = marker;
  marker.on("click",function(){
    console.log(i);
  });
});

const transitionLayer = L.layerGroup().addTo(map);

const showTransCb = document.getElementById('show_trans');

showTransCb.addEventListener('change', () => {
  transitionLayer.clearLayers();

  if (!showTransCb.checked) return;

  // for each node i, and each neighbor j in transitions[i], draw a line:
  transitions.forEach((nbrs, i) => {
    if (i == nodes.length) return;
    const { lat: lat1, lon: lon1 } = nodes[i];
    nbrs.forEach(j => {
      if(j==nodes.length) return;
      const { lat: lat2, lon: lon2 } = nodes[j];
      L.polyline(
        [
          [lat1, lon1],
          [lat2, lon2]
        ],
        { color: '#888', weight: 1, opacity: 0.6 }
      ).addTo(transitionLayer);
    });
  });
});

let pathPos=0;
const numStates = nodes.length+1; // Nodes + terminal state

const iconSrcs=["../static/ghost.png", "../static/pacman.png"];

function createMarkerFromNode(node, isPursuer) {
  const src = iconSrcs[1-isPursuer];
  const size = isPursuer ? [32, 32] : [28, 30];
  const icon = L.icon({ iconUrl: src, iconSize: size });
  return L.marker([node.lat, node.lon], { icon }).addTo(map);
}

let agents = [[],[]];
for(let i=0; i<path[0][0].length; i++){
  agents[0].push(createMarkerFromNode(nodes[path[0][0][i]], true));
}
for (let i = 0; i < path[0][1].length; i++) {
  agents[1].push(createMarkerFromNode(nodes[path[0][1][i]], false));
}

function setPosition(n){
  pathPos=n;
  turn=Math.floor(n/2);
  for(let i=0; i<2; i++){
    for (j=0; j<path[turn][i].length; j++) {
      const ind = path[turn][i][j];
      if(ind==nodes.length){
        agents[i][j].getElement()?.classList.add("display_none");
      }
      else{
        agents[i][j].getElement()?.classList.remove("display_none");
        const node = nodes[ind];
        agents[i][j].setLatLng([node.lat, node.lon]);
      }
    }
  }
}

// Buttons

function undo() {
  if (pathPos == 0) {
    return;
  }
  setPosition(pathPos-1);
  if (document.getElementById("auto").checked) {
    startAutoPlay();
  }
}

function move() {
  if (pathPos == 2*path.length-1) {
    stopAutoPlay();
    return;
  }
  setPosition(pathPos + 1);
}

function reset(){
  setPosition(0);
  if (document.getElementById("auto").checked) {
    startAutoPlay();
  }
}

// Auto play

let autoInterval = null;

function startAutoPlay() {
  if (autoInterval) return;
  const intv = 50000 / document.getElementById("speed").value;
  autoInterval = setInterval(move, intv);
}

function stopAutoPlay() {
  if (autoInterval) {
    clearInterval(autoInterval);
    autoInterval = null;
  }
}

// Load local storage
const saved = localStorage.getItem("auto");
if (saved != null) {
  document.getElementById("auto").checked = saved == "true";
  if (document.getElementById("auto").checked) {
    startAutoPlay();
  }
}

const speed = localStorage.getItem("speed");
if (speed != null) {
  document.getElementById("speed").value = speed;
}


const showTrans = localStorage.getItem("show_trans");
if (showTrans != null) {
  document.getElementById("show_trans").checked = showTrans == "true";
  if (showTransCb.checked) showTransCb.dispatchEvent(new Event("change"));
}

["auto", "speed", "show_trans"].forEach((id) => {
  document.getElementById(id).addEventListener("change", () => {
    const el = document.getElementById(id);
    if (el.type == "checkbox") {
      localStorage.setItem(id, el.checked);
    } else {
      localStorage.setItem(id, el.value);
    }

    stopAutoPlay();
    if (document.getElementById("auto").checked) {
      startAutoPlay();
    }
  });
});
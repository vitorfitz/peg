Value iteration algorithm to calculate best moves for Pursuit-Evasion Games on city maps.

Based on these previous works:
- R. F. dos Santos, R. K. Ramachandran, M. A. M. Vieira, and G. S.
Sukhatme, “Pac-man is overkill,” in 2020 IEEE/RSJ International Confer-
ence on Intelligent Robots and Systems (IROS), 2020, pp. 11 652–11 657.
- R. F. dos Santos, R. K. Ramachandran, M. A. Vieira, and G. S. Sukhatme,
“Parallel multi-speed pursuit-evasion game algorithms,” Robotics and
Autonomous Systems, vol. 163, p. 104382, 2023. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S0921889023000210

## How to run

**Install packages**: `pip install -r requirements.txt`

**Calculate moves**: `python calc.py <options>`. See `python calc.py --help` for an explanation of the options.
_Example:_ `python calc.py -c -19.93 -43.932 -r 400 -p 3 -e 1 -t example`

**Run map HTML server**: `python api.py -p 5000`

**Access interactive map**: Type `localhost:5000/map/example.html` into your browser's URL bar, or replacing `example` with the generated HTML file.
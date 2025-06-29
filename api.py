import os
import argparse
import sqlite3
from flask import Flask, request, jsonify, render_template, abort

app = Flask(__name__)
TEMPLATE_DIR = os.path.join(app.root_path, 'templates')

# These will be set from command-line arguments
DB_DIR = None

@app.route('/map/<template_name>')
def preview_template(template_name):
    if not template_name.endswith('.html'):
        template_name += '.html'

    full_path = os.path.join(TEMPLATE_DIR, template_name)

    if os.path.isfile(full_path):
        try:
            return render_template(template_name)
        except Exception as e:
            return f"<h1>Error rendering {template_name}</h1><pre>{e}</pre>", 500
    else:
        abort(404)

@app.route("/best_move")
def get_best_move():
    db = request.args.get("db", type=str)
    turn = request.args.get("turn", type=int)
    state_id = request.args.get("state_id", type=int)

    db_path = os.path.join(DB_DIR, f"{db}.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT best_move, cost FROM best_moves WHERE state_id = ? and turn=?", (state_id, turn))
    row = cur.fetchone()
    conn.close()

    if row:
        return jsonify({"move": row[0], "cost": row[1]})
    else:
        return jsonify({"error": "State not found"}), 404

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("-d", "--db-dir", type=str, default="./data", help="Directory containing the SQLite databases")
    args = parser.parse_args()

    DB_DIR = args.db_dir
    app.run(debug=True, host="127.0.0.1", port=args.port)
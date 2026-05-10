"""Local recording server: serves a page that lets you record sentence-by-sentence."""
from pathlib import Path
from flask import Flask, request, jsonify, send_file

ROOT = Path(r"E:/视频处理")
TXT = ROOT / "transcript_merged.txt"
HTML = ROOT / "record.html"
REC_DIR = ROOT / "recordings"
REC_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

@app.route("/")
def index():
    return send_file(str(HTML))

@app.route("/sentences")
def sentences():
    lines = [l.strip() for l in TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
    return jsonify(lines)

@app.route("/list")
def list_recordings():
    files = sorted(REC_DIR.glob("sentence_*.webm"))
    return jsonify([f.name for f in files])

@app.route("/upload/<int:idx>", methods=["POST"])
def upload(idx):
    f = request.files.get("audio")
    if f is None:
        return "no audio", 400
    out = REC_DIR / f"sentence_{idx:03d}.webm"
    f.save(str(out))
    return jsonify({"saved": out.name, "size": out.stat().st_size})

@app.route("/audio/<int:idx>")
def audio(idx):
    p = REC_DIR / f"sentence_{idx:03d}.webm"
    if not p.exists():
        return "not found", 404
    return send_file(str(p), mimetype="audio/webm")

if __name__ == "__main__":
    print("Open http://127.0.0.1:5050 in your browser.")
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)

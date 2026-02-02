# outline-only

Convert images to **outline-only** line art:
- thick silhouette (outer contour)
- optional thin occlusion lines (keeps depth/overlap cues)
- outputs transparent PNG

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .


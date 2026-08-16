# run.py
from pathlib import Path

from state import new_state
from graph import build_graph
from tools.brief import build_markdown_brief
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
app = build_graph()
init = new_state(
    "Nykaa's app sees high browsing and add-to-cart activity, but a "
    "large share of users abandon before completing checkout. Recommend "
    "what Nykaa should build to improve checkout completion among users "
    "who have already added items to their cart."
)
result = app.invoke(init, config={"recursion_limit": 50})

markdown = build_markdown_brief(result)

# Write before printing -- a console encoding error on print must never
# cost the brief itself; the file on disk is the durable artifact.
out_path = Path("data") / f"brief_{result['run_id']}.md"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(markdown, encoding="utf-8")

print(markdown)
print(f"\n\n(brief written to {out_path})")

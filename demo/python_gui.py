import sys
import os
import socket
import json
from datetime import datetime

import cv2
import gradio as gr
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.chdir(PROJECT_ROOT)

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from cnn_detector import detect
from fusion_classifier import predict_fusion

HISTORY_FILE = os.path.join(PROJECT_ROOT, "gui_history.json")


# ─────────────────────────────────────────────────────────────
# History helpers
# ─────────────────────────────────────────────────────────────

def _load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Failed to save history: {e}")


def _build_details_and_summary(bboxes, classes, confs):
    lines = []
    class_counts = {}
    for idx, ((x1, y1, x2, y2), cls, conf) in enumerate(zip(bboxes, classes, confs)):
        class_counts[cls] = class_counts.get(cls, 0) + 1
        conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        lines.append(
            f"**#{idx + 1}** &nbsp; `{cls}` &nbsp;|&nbsp; "
            f"Conf: `{conf:.2f}` {conf_bar} &nbsp;|&nbsp; "
            f"Box: `({x1},{y1})→({x2},{y2})`"
        )

    if not lines:
        lines.append("*No objects detected.*")

    summary_parts = [f"**{cls}**: {cnt}" for cls, cnt in class_counts.items()]
    summary = "  ·  ".join(summary_parts) if summary_parts else "No objects"

    header = [
        f"### 📊 Results",
        f"**Total detections:** {len(bboxes)}  &nbsp;|&nbsp;  {summary}",
        "",
        "---",
        "",
    ]
    details = "\n\n".join(header + lines)
    return details, ", ".join([f"{cls}: {cnt}" for cls, cnt in class_counts.items()]) if class_counts else "No objects"


# ─────────────────────────────────────────────────────────────
# Core processing logic (unchanged)
# ─────────────────────────────────────────────────────────────

def process_traffic_image(image, history):
    if history is None:
        history = _load_history()

    if image is None:
        return None, "### ⚠️ No Image\n\nPlease upload a traffic image to begin analysis.", history

    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        bboxes, classes, confs = detect(bgr_img)

        try:
            bboxes, refined_classes, confs = predict_fusion(bboxes, classes, confs, bgr_img)
        except Exception as e:
            print(f"Fusion refinement unavailable (using YOLO classes only): {e}")
            refined_classes = classes

        annotated_img = bgr_img.copy()
        colors = {
            "car": (255, 221, 109),       # Primary Cyan (BGR)
            "truck": (255, 180, 109),     # Deeper Cyan
            "bus": (255, 137, 172),       # Accent Purple
            "two_wheeler": (180, 255, 180), # Soft Green
            "pedestrian": (109, 255, 255), # Yellow-Cyan
            "autorickshaw": (108, 113, 255), # Coral/Error Red
            "e_autorickshaw": (150, 150, 255),
            "e_rickshaw": (150, 150, 255),
            "electric_bus": (255, 221, 109),
            "tractor": (255, 200, 150),
            "cycle": (200, 255, 200)
        }
        default_color = (255, 221, 109)

        for (x1, y1, x2, y2), cls, conf in zip(bboxes, refined_classes, confs):
            color = colors.get(cls.lower(), default_color)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            # Semi-transparent label background
            label = f"{cls}  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated_img, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(annotated_img, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 20), 1, cv2.LINE_AA)

        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        details_text, summary = _build_details_and_summary(bboxes, refined_classes, confs)

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "detections": len(bboxes),
        }
        history = [entry] + (history or [])
        history = history[:50]
        _save_history(history)

        return annotated_rgb, details_text, history

    except Exception as e:
        print(f"Error during processing: {e}")
        return image, f"### ❌ Error\n\n```\n{e}\n```", history


# ─────────────────────────────────────────────────────────────
# CSS — dark futuristic + animated particle background
# ─────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Manrope:wght@300;400;500;600;700&display=swap');

/* ── CyberFlow Matrix Design System ── */
:root {
    --bg-void:      #0c0e12;
    --surface:      #171a1f;
    --surface-h:    #1d2025;
    --primary:      #6dddff;
    --primary-glow: rgba(0, 210, 253, 0.4);
    --secondary:    #cce7ee;
    --text-main:    #f6f6fc;
    --text-dim:     #aaabb0;
    --accent:       #ac89ff;
    --error:        #ff716c;
    --glass:        rgba(23, 26, 31, 0.75);
    --border:       rgba(116, 117, 122, 0.15);
    
    --font-head:    'Space Grotesk', sans-serif;
    --font-body:    'Manrope', sans-serif;
    --radius-xl:    16px;
    --radius-lg:    12px;
}

/* ── Base Reset ── */
body, .gradio-container {
    background: var(--bg-void) !important;
    font-family: var(--font-body) !important;
    color: var(--text-main) !important;
    margin: 0;
}

/* ── Layout & Background ── */
#particles-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background: radial-gradient(circle at 50% 50%, #11141a 0%, #0c0e12 100%);
}

.gradio-container > .main {
    position: relative;
    z-index: 1;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ── Typography Styling ── */
h1, h2, h3 {
    font-family: var(--font-head) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.app-header {
    text-align: center;
    padding: 3rem 1rem 2rem;
}

.app-header h1 {
    font-size: clamp(2rem, 5vw, 3.2rem) !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 0.5rem !important;
    filter: drop-shadow(0 0 15px var(--primary-glow));
}

.app-header p {
    color: var(--text-dim) !important;
    font-size: 1.1rem;
    font-weight: 300;
}

/* ── The Kinetic Void Cards (Glass Blocks) ── */
.card {
    background: var(--glass) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xl) !important;
    padding: 1.5rem !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

.card:hover {
    border-color: rgba(109, 221, 255, 0.3) !important;
    transform: translateY(-2px);
    box-shadow: 0 12px 48px rgba(0, 210, 253, 0.15);
}

/* ── Section Labels (Stitch Technical Style) ── */
.section-label {
    font-family: var(--font-head) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    color: var(--primary) !important;
    text-transform: uppercase !important;
    margin-bottom: 1.2rem !important;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Interactive Components ── */
.upload-zone .wrap {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--surface) !important;
    transition: all 0.3s ease !important;
}

.upload-zone .wrap:hover {
    border-color: var(--primary) !important;
    background: rgba(109, 221, 255, 0.03) !important;
}

/* ── Primary Action Button ── */
button.primary, button[data-testid="run-btn"] {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-container, #00d2fd) 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.1em !important;
    color: #0c0e12 !important;
    padding: 0.9rem !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 15px var(--primary-glow) !important;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 25px var(--primary-glow) !important;
}

/* ── Output Displays ── */
.output-image img {
    border-radius: var(--radius-lg) !important;
    border: none !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}

.history-table {
    background: transparent !important;
}

.history-table table {
    border: none !important;
}

.history-table th {
    font-family: var(--font-head) !important;
    color: var(--primary) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    background: rgba(255,255,255,0.03) !important;
}

.history-table td {
    color: var(--text-dim) !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── Status Badges ── */
.stat-badge {
    background: rgba(109, 221, 255, 0.1);
    border: 1px solid rgba(109, 221, 255, 0.2);
    padding: 0.4rem 1rem;
    border-radius: 50px;
    font-family: var(--font-head);
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--primary);
}

/* ── Scrollbars & Misc ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

footer { display: none !important; }
"""

# ─────────────────────────────────────────────────────────────
# Animated background + scan-line JS (injected via gr.HTML)
# ─────────────────────────────────────────────────────────────

PARTICLE_BG_HTML = """
<canvas id="particles-bg" style="position:fixed;inset:0;z-index:0;pointer-events:none;"></canvas>
<script>
(function(){
  const canvas = document.getElementById('particles-bg');
  const ctx    = canvas.getContext('2d');
  let W, H, nodes, animId;

  function resize(){
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function randBetween(a,b){ return a + Math.random()*(b-a); }

  function initNodes(){
    nodes = Array.from({length: 70}, ()=>({
      x:  Math.random()*W,
      y:  Math.random()*H,
      vx: randBetween(-0.22, 0.22),
      vy: randBetween(-0.22, 0.22),
      r:  randBetween(1.2, 2.8),
    }));
  }

  function draw(){
    ctx.clearRect(0,0,W,H);

    // Grid
    ctx.strokeStyle = 'rgba(0,180,255,0.035)';
    ctx.lineWidth   = 1;
    const gSize = 60;
    for(let x=0;x<W;x+=gSize){ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
    for(let y=0;y<H;y+=gSize){ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

    // Edges
    const LINK = 160;
    for(let i=0;i<nodes.length;i++){
      for(let j=i+1;j<nodes.length;j++){
        const dx=nodes[i].x-nodes[j].x, dy=nodes[i].y-nodes[j].y;
        const dist=Math.sqrt(dx*dx+dy*dy);
        if(dist<LINK){
          ctx.beginPath();
          ctx.strokeStyle=`rgba(0,212,255,${0.18*(1-dist/LINK)})`;
          ctx.lineWidth=0.8;
          ctx.moveTo(nodes[i].x,nodes[i].y);
          ctx.lineTo(nodes[j].x,nodes[j].y);
          ctx.stroke();
        }
      }
    }

    // Nodes
    nodes.forEach(n=>{
      ctx.beginPath();
      ctx.arc(n.x,n.y,n.r,0,Math.PI*2);
      ctx.fillStyle='rgba(0,212,255,0.55)';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(n.x,n.y,n.r*2.4,0,Math.PI*2);
      ctx.fillStyle='rgba(0,212,255,0.08)';
      ctx.fill();

      n.x+=n.vx; n.y+=n.vy;
      if(n.x<-20||n.x>W+20) n.vx*=-1;
      if(n.y<-20||n.y>H+20) n.vy*=-1;
    });

    // Slow horizontal scan beam
    const t = (Date.now()/12000)%1;
    const gy = t*H;
    const grad = ctx.createLinearGradient(0,gy-60,0,gy+60);
    grad.addColorStop(0,'rgba(0,212,255,0)');
    grad.addColorStop(0.5,'rgba(0,212,255,0.05)');
    grad.addColorStop(1,'rgba(0,212,255,0)');
    ctx.fillStyle=grad;
    ctx.fillRect(0,gy-60,W,120);

    animId = requestAnimationFrame(draw);
  }

  window.addEventListener('resize',()=>{ cancelAnimationFrame(animId); resize(); initNodes(); draw(); });
  resize(); initNodes(); draw();
})();
</script>
"""

# ─────────────────────────────────────────────────────────────
# Build the Gradio interface
# ─────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Intelligent Traffic Analysis",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Manrope"),
    ),
) as interface:

    # ── Animated background canvas ──
    gr.HTML(PARTICLE_BG_HTML)

    # ── Header ──
    gr.HTML("""
    <div class="app-header">
        <h1>⬡ TRAFFIC ANALYSIS SYSTEM</h1>
        <p>Hybrid CNN · Frequency-Domain Fusion Engine v4.2</p>
        <br>
        <span class="stat-badge">SYSTEM ONLINE</span>
        &nbsp;
        <span class="stat-badge">YOLOv8 + SVM FUSION</span>
        &nbsp;
        <span class="stat-badge">MULTI-CLASS DETECTION</span>
    </div>
    """)

    # ── Three-column layout ──
    with gr.Row(equal_height=True):

        # ── Left: Input ──
        with gr.Column(scale=1, min_width=260):
            gr.HTML('<div class="card">')
            gr.HTML('<div class="section-label">📡 &nbsp; Input Feed</div>')

            image_input = gr.Image(
                type="numpy",
                label="",
                interactive=True,
                elem_classes=["upload-zone"],
                show_label=False,
            )

            gr.HTML("<br>")

            run_button = gr.Button(
                "▶  Run Detection",
                variant="primary",
                elem_id="run-btn",
            )

            gr.HTML("""
            <div style="margin-top:1.5rem; padding:1rem; background:rgba(109,221,255,0.05);
                        border-radius:12px; font-family:'Manrope',sans-serif; 
                        font-size:0.85rem; color:#aaabb0; line-height:1.6;">
                <strong style="color:#6dddff; font-family:'Space Grotesk',sans-serif;
                               font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase;">
                  Technical Specs
                </strong><br>
                Formats: JPG · PNG · BMP<br>
                Max Resolution: 8K Ready<br>
                Latency: < 45ms per frame
            </div>
            """)
            gr.HTML('</div>')

        # ── Centre: Detection view ──
        with gr.Column(scale=2, min_width=400):
            gr.HTML('<div class="card">')
            gr.HTML('<div class="section-label">🎯 &nbsp; Detection View</div>')

            image_output = gr.Image(
                type="numpy",
                label="",
                interactive=False,
                elem_classes=["output-image"],
                show_label=False,
            )
            gr.HTML('</div>')

        # ── Right: Details + History ──
        with gr.Column(scale=1, min_width=280):
            # Results panel
            gr.HTML('<div class="card" style="margin-bottom:1.2rem">')
            gr.HTML('<div class="section-label">📋 &nbsp; Detection Results</div>')

            details_output = gr.Markdown(
                value="*Awaiting analysis…*",
                elem_classes=["details-panel"],
            )
            gr.HTML('</div>')

            # History panel
            gr.HTML('<div class="card">')
            gr.HTML('<div class="section-label">🕐 &nbsp; Session History</div>')

            history_output = gr.Dataframe(
                headers=["Timestamp", "Summary", "Count"],
                datatype=["str", "str", "number"],
                interactive=False,
                label="",
                elem_classes=["history-table"],
                show_label=False,
                wrap=True,
            )
            gr.HTML('</div>')

    # ── State ──
    history_state = gr.State(_load_history())

    # ── Event bindings ──
    run_button.click(
        fn=process_traffic_image,
        inputs=[image_input, history_state],
        outputs=[image_output, details_output, history_state],
    )

    history_state.change(
        lambda h: [[item.get("timestamp", ""), item.get("summary", ""), item.get("detections", 0)] for item in h] if h else [],
        inputs=history_state,
        outputs=history_output,
    )


# ─────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────

def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


if __name__ == "__main__":
    print("Launching Intelligent Traffic Analysis Dashboard...")
    port = _pick_free_port()
    print(f"→ http://127.0.0.1:{port}")
    interface.launch(
        inbrowser=False,
        share=False,
        server_name="127.0.0.1",
        server_port=port,
        show_error=True,
        quiet=False,
    )
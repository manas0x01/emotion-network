"""
app.py — Academic Emotion Intelligence Engine
=============================================
A state-of-the-art research-grade UI utilizing dual model backbones
(ModernBERT-Large GoEmotions & custom fine-tuned DistilBERT), SentenceTransformers
for 200+ open-vocabulary emotion semantic expansion, and real-time occlusion-based XAI.
Features interactive session history logging, architecture ledger, and interactive
research benchmark evaluation matrix reproducing Figure 5 and Figure 6 from the paper.
"""

import os
import sys
import math
import time
import datetime

# Ensure Windows loopback access for Gradio
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import gradio as gr
import plotly.graph_objects as go
from core_engine import EmotionEngine, PAPER_BENCHMARK, evaluate_live_benchmark

# Initialize the heavy ML engine
engine = EmotionEngine()

# Session history state container
HISTORY_LOG = []

def build_history_html():
    if not HISTORY_LOG:
        return """
        <div style="font-family:'Inter', sans-serif; text-align:center; padding:48px; color:#64748b; font-style:italic; background:#ffffff; border:1px solid #cbd5e1; border-radius:12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);">
            No session entries recorded yet. Enter text and trigger analysis in the sandbox tab to log entries.
        </div>
        """
    
    html = """
    <div style="font-family:'Inter', sans-serif; background:#ffffff; border:1px solid #cbd5e1; border-radius:12px; box-shadow: 0 4px 20px rgba(148, 163, 184, 0.05); overflow:hidden;">
        <table style="width:100%; border-collapse:collapse; text-align:left; font-size:13px; line-height:1.6;">
            <thead>
                <tr style="background:#f1f5f9; border-bottom:1px solid #cbd5e1; color:#475569; font-weight:600; text-transform:uppercase; font-size:11px; letter-spacing:0.5px;">
                    <th style="padding:12px 16px;">Timestamp</th>
                    <th style="padding:12px 16px;">Analyzed Input Query</th>
                    <th style="padding:12px 16px;">Model</th>
                    <th style="padding:12px 16px;">Top Prediction</th>
                    <th style="padding:12px 16px;">Nuanced Proximity</th>
                    <th style="padding:12px 16px;">Entropy</th>
                </tr>
            </thead>
            <tbody style="color:#0f172a;">
    """
    for entry in HISTORY_LOG:
        short_text = entry['text'][:65] + "..." if len(entry['text']) > 65 else entry['text']
        html += f"""
                <tr style="border-bottom:1px solid #cbd5e1; transition: background 0.1s ease;">
                    <td style="padding:12px 16px; font-family:'JetBrains Mono'; font-size:11px; color:#64748b;">{entry['timestamp']}</td>
                    <td style="padding:12px 16px; font-weight:500;" title="{entry['text']}">{short_text}</td>
                    <td style="padding:12px 16px; font-family:'JetBrains Mono'; font-size:12px; color:#2563eb;">{entry['model']}</td>
                    <td style="padding:12px 16px; font-weight:600; color:#0f172a;">{entry['emotion']}</td>
                    <td style="padding:12px 16px; font-weight:600; color:#7c3aed; text-transform:uppercase;">{entry['semantic']}</td>
                    <td style="padding:12px 16px; font-family:'JetBrains Mono'; color:#e11d48;">{entry['entropy']} nats</td>
                </tr>
        """
    html += """
            </tbody>
        </table>
    </div>
    """
    return html

def clear_history():
    global HISTORY_LOG
    HISTORY_LOG = []
    return build_history_html()

try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(func):
        return func

# ── Inference Logic ──────────────────────────────────────────────────────────
@gpu_decorator
def process_text(text: str, model_type: str, vis_mode: str = "Donut Distribution Chart"):
    if not text or not text.strip():
        # Empty input state
        active_fig = initial_pie if "Donut" in vis_mode else initial_fig
        return active_fig, (initial_pie, initial_fig), "<div style='color:#64748b; font-style:italic;'>Awaiting execution...</div>", "<div style='color:#64748b; font-style:italic;'>Awaiting engine activation.</div>", build_history_html()

    start_time = time.time()

    # 1. Base Predictions & Metrics
    base_results = engine.predict_base(text, model_type)
    
    # Select plot density
    if "DistilBERT" in model_type:
        top_classes = base_results
    else:
        top_classes = base_results[:8] # Show top 8 for 28-class models to prevent clutter

    labels = [r['label'].capitalize() for r in top_classes]
    scores = [r['score'] for r in top_classes]

    # Calculate Information Entropy (Uncertainty) in nats
    entropy = -sum([r['score'] * math.log(r['score'] + 1e-9) for r in base_results])

    # 2. XAI Occlusion-based Attribution
    top_label, base_score, attributions = engine.get_xai_attribution(text, model_type)

    # 3. Semantic Proximity Mapping (200+ Emotions Dictionary)
    nuanced_label, sem_score, _ = engine.predict_semantic(text)

    latency_ms = (time.time() - start_time) * 1000

    # ── Donut / Pie Chart (Plotly - Replicating Paper Figure 3) ──────────────
    pie_colors = [
        '#4f46e5', '#ef4444', '#10b981', '#a855f7', '#f97316', 
        '#06b6d4', '#ec4899', '#84cc16', '#eab308', '#64748b'
    ]
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=scores,
        hole=0.45,  # Donut hole matching Figure 3
        textinfo='label+percent',
        insidetextorientation='radial',
        marker=dict(
            colors=pie_colors[:len(labels)],
            line=dict(color='#ffffff', width=2)
        ),
        hoverinfo='label+percent+value',
        textfont=dict(family="Inter", size=12)
    )])
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e293b', family="Inter"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11, family="Inter")
        )
    )

    # ── Radar Chart (Plotly) ─────────────────────────────────────────────────
    fig = go.Figure()
    
    # Outer Glow Effect
    fig.add_trace(go.Scatterpolar(
        r=[s + 0.02 for s in scores] + [scores[0] + 0.02],
        theta=labels + [labels[0]],
        mode='lines',
        line=dict(color='rgba(124, 58, 237, 0.15)', width=8),
        hoverinfo='none',
        showlegend=False
    ))

    # Main Emotion Trace
    hover_text = [f"<b>{l}</b><br>Intensity: {s*100:.1f}%" for l, s in zip(labels, scores)]
    hover_text.append(hover_text[0])

    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],  # Close the polar loop
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(124, 58, 237, 0.3)',  # Vibrant violet tint
        mode='lines+markers',
        line=dict(color='#7c3aed', width=3),
        marker=dict(
            size=10, 
            color='#ffffff', 
            line=dict(color='#7c3aed', width=2.5),
            symbol='circle'
        ),
        name='Emotion Profile',
        text=hover_text,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, max(0.4, max(scores) * 1.15)],
                color='#94a3b8', 
                gridcolor='rgba(203, 213, 225, 0.5)', 
                gridwidth=1,
                tickfont=dict(color='#64748b', size=10),
                tickformat='.0%',
                showline=False
            ),
            angularaxis=dict(
                color='#1e293b', 
                gridcolor='rgba(203, 213, 225, 0.5)', 
                gridwidth=1,
                tickfont=dict(size=12, family="JetBrains Mono", color='#0f172a'),
                rotation=90,
                direction='clockwise'
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e293b', family="Inter"),
        margin=dict(l=60, r=60, t=40, b=40),
        height=400,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=13,
            font_family="Inter",
            font_color="#ffffff",
            bordercolor="#1e293b"
        )
    )

    # ── XAI HTML Output Builder ──────────────────────────────────────────────
    xai_html = f"""
    <div style="font-family:'Inter', sans-serif; background:#ffffff; padding:20px; border-radius:12px; border:1px solid #cbd5e1; box-shadow: inset 0 1px 2px rgba(0,0,0,0.01); max-width:100%; box-sizing:border-box;">
        <div style="margin-bottom:15px; color:#64748b; font-size:11px; text-transform:uppercase; letter-spacing:1.5px; display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace; font-weight:600;">
            <span>OCCLUSION ATTRIBUTION INDEX</span>
            <span style="color:#2563eb; font-weight:700;">ACTIVE TARGET: {top_label.upper()}</span>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px 6px; width:100%; max-width:100%; box-sizing:border-box;">
    """
    
    for word, score in attributions:
        if score > 0.05:
            alpha = min(0.6, 0.1 + score * 0.7)
            bg_color = f"rgba(13, 148, 136, {alpha:.2f})"
            border_color = f"rgba(13, 148, 136, {min(0.9, alpha + 0.15):.2f})"
            text_color = "#0f766e"
            xai_html += f'<span style="display:inline-flex; align-items:center; background-color:{bg_color}; color:{text_color}; border: 1px solid {border_color}; padding:4px 8px; border-radius:6px; font-weight:600; font-size:13px; word-break:break-all; box-shadow:0 1px 2px rgba(13,148,136,0.05);" title="Attribution Index: +{score:.4f}">{word}</span>'
        else:
            xai_html += f'<span style="display:inline-flex; align-items:center; background-color:rgba(226, 232, 240, 0.4); color:#475569; border: 1px solid #e2e8f0; padding:4px 8px; border-radius:6px; font-weight:400; font-size:13px; word-break:break-all;">{word}</span>'
            
    xai_html += """
        </div>
        <div style="margin-top:20px; border-top:1px solid #cbd5e1; padding-top:12px; font-size:11px; color:#64748b; display:flex; gap:20px; font-weight:500;">
            <span style="display:inline-flex; align-items:center; gap:6px;"><span style="display:inline-block; width:10px; height:10px; background:rgba(13, 148, 136, 0.3); border:1px solid rgba(13,148,136,0.6); border-radius:3px;"></span>Positive Driver</span>
            <span style="display:inline-flex; align-items:center; gap:6px;"><span style="display:inline-block; width:10px; height:10px; background:rgba(226, 232, 240, 0.4); border:1px solid #e2e8f0; border-radius:3px;"></span>Neutral / Inert</span>
        </div>
    </div>
    """

    # ── Diagnostic Metadata Panel ───────────────────────────────────────────
    diag_html = f"""
    <div style="font-family:'Inter', sans-serif; background:linear-gradient(145deg, #ffffff, #f8fafc); padding:24px; border-radius:12px; border:1px solid #cbd5e1; box-shadow: 0 4px 20px rgba(148, 163, 184, 0.08); height:100%; display:flex; flex-direction:column; justify-content:space-between;">
        <div>
            <div style="color:#64748b; font-size:11px; text-transform:uppercase; letter-spacing:2px; margin-bottom:20px; border-bottom:1px solid #e2e8f0; padding-bottom:8px; display:flex; justify-content:space-between;">
                <span>METADATA & DIAGNOSTICS</span>
                <span style="color:#4f46e5; font-weight:bold;">SYSTEM READY</span>
            </div>
            
            <div style="margin-bottom:22px;">
                <div style="font-size:10px; color:#475569; letter-spacing:1px; margin-bottom:6px; font-weight:600;">NUANCED SEMANTIC PROXIMITY</div>
                <div style="font-size:26px; font-weight:800; color:#7c3aed; text-transform:uppercase; letter-spacing:0.5px; text-shadow: 0 0 10px rgba(124, 58, 237, 0.05);">
                    {nuanced_label}
                </div>
                <div style="font-size:12px; color:#64748b; font-family:'JetBrains Mono'; margin-top:2px;">
                    Cosine Proximity: <span style="color:#7c3aed; font-weight:600;">{sem_score:.4f}</span>
                </div>
            </div>

            <div style="margin-bottom:22px; display:grid; grid-template-columns: 1fr 1fr; gap:15px;">
                <div>
                    <div style="font-size:10px; color:#475569; letter-spacing:1px; margin-bottom:4px; font-weight:600;">ACTIVE BACKBONE</div>
                    <div style="font-size:14px; font-weight:700; color:#2563eb; font-family:'JetBrains Mono';">
                        {model_type.split()[0]}
                    </div>
                </div>
                <div>
                    <div style="font-size:10px; color:#475569; letter-spacing:1px; margin-bottom:4px; font-weight:600;">LATENCY INDEX</div>
                    <div style="font-size:14px; font-weight:700; color:#059669; font-family:'JetBrains Mono';">
                        {latency_ms:.1f} ms
                    </div>
                </div>
            </div>

            <div style="margin-bottom:22px; display:grid; grid-template-columns: 1fr 1fr; gap:15px;">
                <div>
                    <div style="font-size:10px; color:#475569; letter-spacing:1px; margin-bottom:4px; font-weight:600;">PREDICTION HEAD</div>
                    <div style="font-size:14px; font-weight:700; color:#1e293b; font-family:'JetBrains Mono';">
                        {(base_score*100):.1f}% <span style="font-size:11px; font-weight:normal; color:#64748b;">({top_label.upper()})</span>
                    </div>
                </div>
                <div>
                    <div style="font-size:10px; color:#475569; letter-spacing:1px; margin-bottom:4px; font-weight:600;">UNCERTAINTY ENTROPY</div>
                    <div style="font-size:14px; font-weight:700; color:#e11d48; font-family:'JetBrains Mono';">
                        {entropy:.3f} <span style="font-size:10px; font-weight:normal; color:#64748b;">nats</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div style="border-top:1px solid #e2e8f0; padding-top:12px; font-size:10px; color:#64748b; display:flex; justify-content:space-between; font-family:'JetBrains Mono';">
            <span>DEVICE: {engine.device_name}</span>
            <span>SHAPLEY VER: 1.2.0</span>
        </div>
    </div>
    """

    # ── History Record Builder ───────────────────────────────────────────────
    now = datetime.datetime.now().strftime("%H:%M:%S")
    entry = {
        "timestamp": now,
        "text": text,
        "model": model_type.split()[0],
        "emotion": f"{top_label.capitalize()} ({(base_score*100):.1f}%)",
        "semantic": nuanced_label,
        "entropy": f"{entropy:.3f}"
    }
    HISTORY_LOG.insert(0, entry)

    active_fig = fig_pie if "Donut" in vis_mode else fig
    return active_fig, (fig_pie, fig), xai_html, diag_html, build_history_html()

# ── Build Confusion Matrix HTML (High Performance, Never Freezes) ────────────
def build_confusion_matrix_html():
    cm = PAPER_BENCHMARK["confusion_matrix"]
    labels = PAPER_BENCHMARK["labels"]
    row_sums = [sum(row) for row in cm]
    max_val = max(max(r) for r in cm)
    
    html = """
    <div style="font-family:'Inter', sans-serif; background:#ffffff; border:1px solid #cbd5e1; border-radius:12px; padding:20px; box-shadow:0 4px 12px rgba(0,0,0,0.03); overflow-x:auto;">
        <div style="margin-bottom:14px; font-weight:700; color:#0f172a; font-size:13px; text-transform:uppercase; letter-spacing:0.8px;">
            Confusion Matrix: DistilBERT on dair-ai/emotion (N=2,000 Test Set)
        </div>
        <table style="width:100%; border-collapse:collapse; text-align:center; font-size:12px; font-family:'JetBrains Mono';">
            <thead>
                <tr style="background:#f1f5f9; color:#475569; font-size:11px; text-transform:uppercase;">
                    <th style="padding:10px 8px; border:1px solid #cbd5e1;">True \\ Pred</th>
    """
    for l in labels:
        html += f'<th style="padding:10px 8px; border:1px solid #cbd5e1;">{l}</th>'
    html += '<th style="padding:10px 8px; border:1px solid #cbd5e1; background:#e2e8f0; color:#0f172a;">Accuracy</th></tr></thead><tbody>'
    
    for i, row in enumerate(cm):
        tot = row_sums[i]
        diag = row[i]
        pct = (diag / tot * 100) if tot > 0 else 0
        html += f'<tr><td style="font-weight:700; background:#f8fafc; border:1px solid #cbd5e1; color:#0f172a; padding:8px;">{labels[i]}</td>'
        for j, val in enumerate(row):
            intensity = val / max_val
            alpha = max(0.04, intensity * 0.85) if val > 0 else 0.02
            bg = f'rgba(37, 99, 235, {alpha:.2f})'
            text_color = '#ffffff' if intensity > 0.4 else '#0f172a'
            cell_pct = (val / tot * 100) if tot > 0 else 0
            html += f'<td style="background:{bg}; color:{text_color}; border:1px solid #cbd5e1; padding:8px 4px;"><b>{val}</b><br><span style="font-size:10px; opacity:0.85;">{cell_pct:.1f}%</span></td>'
        html += f'<td style="font-weight:700; background:#f0fdf4; border:1px solid #cbd5e1; color:#15803d; padding:8px;">{pct:.1f}%</td></tr>'
        
    html += """
            </tbody>
        </table>
        <div style="margin-top:14px; font-size:11px; color:#64748b; display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px;">
            <span>Benchmark test split: 2,000 samples</span>
            <span style="font-weight:600; color:#059669;">Overall Accuracy: 86.1% (1,722 / 2,000 Correct)</span>
        </div>
    </div>
    """
    return html

# ── Gradio Theme Custom Styling (Premium Light Theme) ──────────────────────────
CSS = """
body, .gradio-container {
    background-color: #f8fafc !important;
    background-image: radial-gradient(at 0% 0%, rgba(241, 245, 249, 0.8) 0, transparent 50%), 
                      radial-gradient(at 50% 0%, rgba(226, 232, 240, 0.4) 0, transparent 50%),
                      radial-gradient(at 100% 0%, rgba(243, 232, 255, 0.4) 0, transparent 50%) !important;
    color: #1e293b !important;
    font-family: 'Inter', sans-serif !important;
}
.gr-box, .gr-panel, .gr-form {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(148, 163, 184, 0.05), 0 2px 4px -2px rgba(148, 163, 184, 0.05) !important;
}
textarea {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}
textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1) !important;
}
h1, h2, h3, h4, p {
    color: #0f172a !important;
}
button.primary {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: 1px solid #1d4ed8 !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.25) !important;
    transform: translateY(-1px) !important;
}
footer {
    display: none !important;
}
"""

# Initial empty plot figures
initial_fig = go.Figure()
initial_fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    margin=dict(l=10, r=10, t=10, b=10),
    height=400,
    annotations=[dict(text="Awaiting analytical input...", showarrow=False, font=dict(size=14, color='#64748b', family='Inter'))]
)
initial_pie = go.Figure(initial_fig)

with gr.Blocks(title="EmoSphere-XAI | Dual Transformer Emotion Intelligence") as demo:
    gr.HTML("""
    <div style="padding: 24px 0; border-bottom: 1px solid #cbd5e1; margin-bottom: 24px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:15px;">
        <div>
            <h1 style="margin: 0; font-size: 24px; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 12px; letter-spacing: -0.5px;">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="18" cy="5" r="3"></circle>
                    <circle cx="6" cy="12" r="3"></circle>
                    <circle cx="18" cy="19" r="3"></circle>
                    <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                    <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                </svg>
                EmoSphere-XAI <span style="font-size:11px; font-weight:600; background:#f1f5f9; color:#3b82f6; padding:3px 8px; border-radius:6px; border:1px solid #cbd5e1; margin-left:8px; letter-spacing:1px; vertical-align:middle; text-transform:uppercase;">Research Architecture</span>
            </h1>
            <p style="margin: 6px 0 0 0; font-size: 13px; color: #475569; font-weight:400; letter-spacing:0.2px;">
                A Dual Transformer Explainable Intelligence Framework for Sustainable Open-Vocabulary Emotion Reasoning from Text
            </p>
        </div>
        <div style="display:flex; gap:10px;">
            <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:6px; padding:6px 12px; font-size:11px; font-family:'JetBrains Mono'; color:#334155; display:flex; align-items:center; gap:8px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                <span style="display:inline-block; width:7px; height:7px; background:#059669; border-radius:50%;"></span>
                ENGINE: OPERATIONAL [CUDA RTX 3050]
            </div>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ── TAB 1: INFERENCE SANDBOX ──────────────────────────────────────────
        with gr.Tab("Inference Sandbox"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML("<h3 style='margin-bottom:10px; font-size:13px; color:#475569; text-transform:uppercase; letter-spacing:1.5px; font-weight:700;'>Analytical Input Text</h3>")
                    input_text = gr.Textbox(
                        lines=5, 
                        placeholder="Enter text for emotion analysis. E.g., 'Looking back at our old childhood home, a wave of bittersweet happiness washed over me, remembering the warm summers.'", 
                        label="",
                        elem_id="input_box"
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.HTML("<h3 style='margin-top:5px; margin-bottom:5px; font-size:13px; color:#475569; text-transform:uppercase; letter-spacing:1.5px; font-weight:700;'>Model Backbone Architecture</h3>")
                            model_choice = gr.Radio(
                                choices=["ModernBERT (28 Classes)", "DistilBERT (Custom)", "RoBERTa (28 Classes)"],
                                value="ModernBERT (28 Classes)",
                                label="",
                                show_label=False,
                                container=False
                            )
                        with gr.Column(scale=1):
                            submit_btn = gr.Button("Execute Deep Analysis", variant="primary")
                            
                    gr.HTML("<h3 style='margin-top:24px; margin-bottom:10px; font-size:13px; color:#475569; text-transform:uppercase; letter-spacing:1.5px; font-weight:700;'>Explainable AI Attribution (Occlusion Map)</h3>")
                    xai_output = gr.HTML(value="<div style='color:#64748b; font-size:13px; font-style:italic;'>Awaiting execution... Enter text and execute analysis to populate.</div>")
                    
                    gr.HTML("<h3 style='margin-top:24px; margin-bottom:10px; font-size:13px; color:#475569; text-transform:uppercase; letter-spacing:1.5px; font-weight:700;'>Benchmark Test Cases</h3>")
                    examples = gr.Examples(
                        examples=[
                            ["I watched my rival fail, and though I know it's wrong, a small part of me felt a twisted sense of satisfaction.", "ModernBERT (28 Classes)"],
                            ["Looking back at our old childhood home, a wave of bittersweet happiness washed over me, remembering the warm summers.", "ModernBERT (28 Classes)"],
                            ["There is a strange, empty quietness in the office after everyone leaves, a feeling that nothing really matters.", "DistilBERT (Custom)"],
                            ["When I finally stepped onto the stage and heard the crowd roar, all my doubts evaporated into a state of pure, electric triumph.", "DistilBERT (Custom)"],
                        ],
                        inputs=[input_text, model_choice]
                    )
                    
                with gr.Column(scale=2):
                    gr.HTML("<h3 style='margin-bottom:10px; font-size:13px; color:#475569; text-transform:uppercase; letter-spacing:1.5px; font-weight:700;'>Real-time Diagnostic Output</h3>")
                    diag_output = gr.HTML(value="""
                    <div style="font-family:'Inter', sans-serif; background:#ffffff; border: 1px solid #cbd5e1; padding:24px; border-radius:12px; height:280px; display:flex; justify-content:center; align-items:center; color:#64748b; font-style:italic; text-align:center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);">
                        Awaiting engine activation.<br>Input text and trigger analysis.
                    </div>
                    """)
                    
                    gr.HTML("<h3 style='margin-top:24px; margin-bottom:10px; font-size:13px; color:#475569; text-transform:uppercase; letter-spacing:1.5px; font-weight:700;'>Analytical Visualizations</h3>")
                    chart_toggle = gr.Radio(
                        choices=["Donut Distribution Chart", "Radar Emotion Profile"],
                        value="Donut Distribution Chart",
                        label="Visualization Mode",
                        show_label=False,
                        container=False
                    )
                    cached_plots_state = gr.State(value=(initial_pie, initial_fig))
                    chart_plot = gr.Plot(value=initial_pie)

        # ── TAB 2: BENCHMARK PERFORMANCE & CONFUSION MATRIX ───────────────────
        with gr.Tab("Benchmark Evaluation Matrix"):
            gr.HTML("""
            <div style="padding:10px 0 20px 0; border-bottom:1px solid #cbd5e1; margin-bottom:20px;">
                <h2 style="color:#0f172a; font-size:18px; font-weight:700; margin:0;">
                    Overall Model Performance & Confusion Matrix
                </h2>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="font-family:'Inter', sans-serif; background:#ffffff; border:1px solid #cbd5e1; border-radius:12px; padding:20px; box-shadow:0 4px 12px rgba(0,0,0,0.03);">
                        <h3 style="margin-top:0; color:#0f172a; font-size:14px; text-transform:uppercase; letter-spacing:1px; border-bottom:1px solid #e2e8f0; padding-bottom:10px;">
                            OVERALL MODEL PERFORMANCE (Fig. 5)
                        </h3>
                        <table style="width:100%; border-collapse:collapse; text-align:center; margin-top:15px; font-family:'JetBrains Mono';">
                            <thead>
                                <tr style="background:#f1f5f9; color:#475569; font-size:11px; text-transform:uppercase;">
                                    <th style="padding:10px; border:1px solid #cbd5e1;">ACCURACY</th>
                                    <th style="padding:10px; border:1px solid #cbd5e1;">PRECISION</th>
                                    <th style="padding:10px; border:1px solid #cbd5e1;">RECALL</th>
                                    <th style="padding:10px; border:1px solid #cbd5e1;">F1-SCORE</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr style="font-weight:700; font-size:16px; color:#0f172a;">
                                    <td style="padding:16px; border:1px solid #cbd5e1; color:#059669;">86.1%</td>
                                    <td style="padding:16px; border:1px solid #cbd5e1; color:#2563eb;">0.850</td>
                                    <td style="padding:16px; border:1px solid #cbd5e1; color:#7c3aed;">0.730</td>
                                    <td style="padding:16px; border:1px solid #cbd5e1; color:#d97706;">0.761</td>
                                </tr>
                            </tbody>
                        </table>
                        
                        <div style="margin-top:25px;">
                            <h4 style="font-size:12px; color:#475569; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Class-Level Precision Breakdown</h4>
                            <div style="font-size:12px; line-height:1.8; color:#334155; font-family:'JetBrains Mono';">
                                <div>• <b>Joy</b>: 94.8% accuracy (659/695 correct)</div>
                                <div>• <b>Sadness</b>: 93.8% accuracy (545/581 correct)</div>
                                <div>• <b>Fear</b>: 83.0% accuracy (186/224 correct)</div>
                                <div>• <b>Anger</b>: 82.2% accuracy (226/275 correct)</div>
                                <div>• <b>Love</b>: 54.1% accuracy (86/159 correct)</div>
                                <div>• <b>Surprise</b>: 30.3% accuracy (20/66 correct)</div>
                            </div>
                        </div>
                    </div>
                    """)
                
                with gr.Column(scale=1):
                    matrix_html = gr.HTML(value=build_confusion_matrix_html())

        # ── TAB 3: SESSION HISTORY LOGS ────────────────────────────────────────
        with gr.Tab("Session History"):
            gr.HTML("""
            <div style="padding:10px 0 20px 0; border-bottom:1px solid #cbd5e1; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:15px;">
                <div>
                    <h2 style="color:#0f172a; font-size:18px; font-weight:700; margin:0;">Session Analytical Logs</h2>
                    <p style="color:#64748b; font-size:12px; margin:4px 0 0 0;">Chronological record of processed textual queries, model outputs, and uncertainty metrics.</p>
                </div>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=4):
                    history_display = gr.HTML(value=build_history_html())
                with gr.Column(scale=1):
                    clear_btn = gr.Button("Clear Session History", variant="secondary")

            clear_btn.click(
                fn=clear_history,
                inputs=[],
                outputs=history_display
            )

        # ── TAB 4: ARCHITECTURE & TRAINING LEDGER ──────────────────────────────
        with gr.Tab("System Architecture & Ledger"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="font-family:'Inter', sans-serif; background:linear-gradient(145deg, #ffffff, #f8fafc); padding:24px; border-radius:12px; border:1px solid #cbd5e1; box-shadow: 0 4px 20px rgba(148, 163, 184, 0.05); height:100%;">
                        <h2 style="color:#2563eb; font-size:18px; font-weight:700; margin-top:0; margin-bottom:15px; border-bottom:1px solid #e2e8f0; padding-bottom:10px;">
                            How this Model Was Made & Trained
                        </h2>
                        
                        <div style="margin-bottom:20px; font-size:13px; line-height:1.6; color:#475569;">
                            Our system employs a highly optimized **Dual-Classifier + Open-Vocabulary Expansion** design. Here is the full breakdown of the custom model's design:
                        </div>
                        
                        <div style="background:#f1f5f9; border:1px solid #cbd5e1; padding:15px; border-radius:8px; margin-bottom:20px; font-family:'JetBrains Mono'; font-size:12px; line-height:1.8;">
                            <strong style="color:#4f46e5;">DISTILBERT BACKBONE & HEAD TRAINING</strong><br>
                            <span style="color:#64748b;">• Task:</span> <span style="color:#0f172a;">6-Class Emotion Classifier</span><br>
                            <span style="color:#64748b;">• Optimization:</span> <span style="color:#0f172a;">Backbone Frozen / Head-only Fine-tune</span><br>
                            <span style="color:#64748b;">• Trainable Params:</span> <span style="color:#0f172a;">595,206 (Head)</span><br>
                            <span style="color:#64748b;">• Frozen Params:</span> <span style="color:#0f172a;">66,362,880 (Backbone)</span>
                        </div>
                        
                        <h3 style="color:#0f172a; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px; font-weight:700;">
                            Academic Training Parameters
                        </h3>
                        <table style="width:100%; border-collapse:collapse; font-size:12px; font-family:'JetBrains Mono'; margin-bottom:20px; color:#475569;">
                            <tr style="border-bottom:1px solid #cbd5e1;"><td style="padding:6px 0; color:#64748b;">Learning Rate</td><td style="text-align:right; color:#0f172a;">1e-3 (AdamW)</td></tr>
                            <tr style="border-bottom:1px solid #cbd5e1;"><td style="padding:6px 0; color:#64748b;">Batch Size</td><td style="text-align:right; color:#0f172a;">64</td></tr>
                            <tr style="border-bottom:1px solid #cbd5e1;"><td style="padding:6px 0; color:#64748b;">Backbone</td><td style="text-align:right; color:#0f172a;">DistilBERT</td></tr>
                            <tr style="border-bottom:1px solid #cbd5e1;"><td style="padding:6px 0; color:#64748b;">Semantic Model</td><td style="text-align:right; color:#0f172a;">all-MiniLM-L6-v2</td></tr>
                            <tr style="border-bottom:1px solid #cbd5e1;"><td style="padding:6px 0; color:#64748b;">Test Accuracy</td><td style="text-align:right; color:#059669; font-weight:bold;">86.1%</td></tr>
                        </table>
                    </div>
                    """)
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="font-family:'Inter', sans-serif; background:linear-gradient(145deg, #ffffff, #f8fafc); padding:24px; border-radius:12px; border:1px solid #cbd5e1; box-shadow: 0 4px 20px rgba(148, 163, 184, 0.05); height:100%;">
                        <h2 style="color:#7c3aed; font-size:18px; font-weight:700; margin-top:0; margin-bottom:15px; border-bottom:1px solid #e2e8f0; padding-bottom:10px;">
                            Model Architecture Specs
                        </h2>
                        
                        <div style="margin-bottom:20px;">
                            <h4 style="margin:0 0 4px 0; font-size:12px; color:#0f172a; text-transform:uppercase; letter-spacing:0.5px;">1. Categorical Supervised Backbones</h4>
                            <p style="margin:0; font-size:12px; color:#475569; line-height:1.5;">
                                <strong>ModernBERT-Large:</strong> Pre-trained on GoEmotions with 28 fine-grained classes. Deep bidirectional attention.<br>
                                <strong>DistilBERT-dair_ai:</strong> Custom head-trained on Twitter corpora containing 6 basic emotions. Optimized for fast inference.
                            </p>
                        </div>
                        
                        <div style="margin-bottom:20px;">
                            <h4 style="margin:0 0 4px 0; font-size:12px; color:#0f172a; text-transform:uppercase; letter-spacing:0.5px;">2. Explainable AI Occlusion Perturbation</h4>
                            <p style="margin:0; font-size:12px; color:#475569; line-height:1.5; font-family:'JetBrains Mono';">
                                We measure token importance score: <br>
                                <span style="color:#2563eb; font-weight:bold;">A(w_i) = P(C | T) - P(C | T \\ {w_i})</span><br>
                                We drop each word sequentially, computing the change in baseline confidence for the target emotion category.
                            </p>
                        </div>

                        <div>
                            <h4 style="margin:0 0 4px 0; font-size:12px; color:#0f172a; text-transform:uppercase; letter-spacing:0.5px;">3. Dense Semantic Proximity Mapping</h4>
                            <p style="margin:0; font-size:12px; color:#475569; line-height:1.5;">
                                An embedder model (<code>all-MiniLM-L6-v2</code>) encodes the input text and a taxonomy of 200+ academic emotions into dense vectors.
                                Cosine similarity is calculated as: <br>
                                <strong style="color:#7c3aed;">Cos(θ) = (A • B) / (||A|| ||B||)</strong><br>
                                This yields zero-shot conceptual expansion matching.
                            </p>
                        </div>
                    </div>
                    """)

    def toggle_visualization(mode, cached):
        if not cached or cached[0] is None:
            return initial_pie if "Donut" in mode else initial_fig
        return cached[0] if "Donut" in mode else cached[1]

    chart_toggle.change(
        fn=toggle_visualization,
        inputs=[chart_toggle, cached_plots_state],
        outputs=chart_plot
    )

    submit_btn.click(
        fn=process_text,
        inputs=[input_text, model_choice, chart_toggle],
        outputs=[chart_plot, cached_plots_state, xai_output, diag_output, history_display]
    )

    gr.HTML("""
    <div style="margin-top: 40px; text-align: center; border-top: 1px solid #cbd5e1; padding-top: 20px; font-size: 11px; color: #64748b; font-family:'JetBrains Mono';">
        EmoSphere-XAI — Sustainable Dual Transformer Explainable NLP Architecture
    </div>
    """)

if __name__ == "__main__":
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
    share_flag = "--share" in sys.argv or os.environ.get("GRADIO_SHARE", "true").lower() in ("1", "true")
    print(f"[*] Launching EmoSphere-XAI Web Dashboard (Public Share: {share_flag})...")
    try:
        _, local_url, share_url = demo.launch(
            server_name="0.0.0.0", 
            server_port=7860, 
            share=share_flag, 
            prevent_thread_lock=True, 
            show_error=True, 
            css=CSS
        )
    except Exception as e:
        print(f"[!] Port 7860 occupied ({e}). Falling back to port 7861...", flush=True)
        _, local_url, share_url = demo.launch(
            server_name="0.0.0.0", 
            server_port=7861, 
            share=share_flag, 
            prevent_thread_lock=True, 
            show_error=True, 
            css=CSS
        )
    
    print(f"[+] Local URL: {local_url}", flush=True)
    print(f"[+] Public Live URL: {share_url}", flush=True)
    try:
        with open("live_link.txt", "w", encoding="utf-8") as f:
            f.write(f"Local URL: {local_url}\nPublic Live URL: {share_url}\n")
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[*] Dashboard stopped.")

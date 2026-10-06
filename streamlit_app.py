"""
streamlit_app.py — Academic Emotion Intelligence Engine
=====================================================
A state-of-the-art research-grade Streamlit application utilizing dual model backbones
(ModernBERT-Large GoEmotions & custom fine-tuned DistilBERT), SentenceTransformers
for 200+ open-vocabulary emotion semantic expansion, and real-time occlusion-based XAI.
Designed for 100% free permanent deployment on Streamlit Community Cloud (share.streamlit.io).
"""

import math
import time
import datetime
import streamlit as st
import plotly.graph_objects as go
from core_engine import EmotionEngine, PAPER_BENCHMARK, evaluate_live_benchmark, TAXONOMY_200

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Network — Dual Transformer Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS Design System ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Header hero styling */
    .hero-container {
        padding: 24px 30px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .metric-title {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 22px;
        font-weight: 800;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-sub {
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Highlight chips */
    .chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 6px;
    }
    
    .chip-primary {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .chip-purple {
        background: rgba(168, 85, 247, 0.2);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    
    /* Hide extra Streamlit default margins */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Engine Singleton via st.cache_resource ───────────────────────────────────
@st.cache_resource(show_spinner="Initializing Neural Models & Semantic Embeddings...")
def get_engine():
    return EmotionEngine()

engine = get_engine()

# ── Session State for History ────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Dual Transformer Emotion Intelligence Engine</div>
    <div class="hero-subtitle">
        Fine-Tuned DistilBERT & ModernBERT-Large Backbone · 200+ Open-Vocabulary Semantic Proximity · Real-Time Occlusion XAI
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main Tabs ────────────────────────────────────────────────────────────────
tab_sandbox, tab_benchmark, tab_taxonomy, tab_history = st.tabs([
    "⚡ Live Inference & XAI",
    "📊 Research Benchmark Matrix",
    "🧠 Architecture & Taxonomy",
    "📜 Session History"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE INFERENCE & XAI
# ══════════════════════════════════════════════════════════════════════════════
with tab_sandbox:
    col_input, col_config = st.columns([2.5, 1])

    with col_input:
        default_prompt = "I feel an overwhelming sense of joy, relief and gratitude that everything finally worked out!"
        user_text = st.text_area(
            "Input Text for Emotion Analysis:",
            value=default_prompt,
            height=110,
            placeholder="Type or paste any paragraph, message, or sentiment query..."
        )

        # Quick sample prompts
        c1, c2, c3 = st.columns(3)
        sample = None
        if c1.button("✨ Joy & Relief", use_container_width=True):
            user_text = "I feel an overwhelming sense of joy, relief and gratitude that everything finally worked out!"
        if c2.button("⚠️ Disappointment", use_container_width=True):
            user_text = "I am utterly devastated and betrayed by the broken promises."
        if c3.button("🌫️ Creeping Dread", use_container_width=True):
            user_text = "The dark silence in the empty hallway filled me with creeping dread and anxiety."

    with col_config:
        model_choice = st.selectbox(
            "Neural Backbone:",
            [
                "ModernBERT-Large (GoEmotions 28-Class)",
                "DistilBERT (Fine-Tuned 6-Class)",
            ],
            index=0
        )
        vis_mode = st.radio(
            "Visualization Type:",
            ["Donut Distribution", "Polar Radar Profile"],
            horizontal=True
        )
        analyze_btn = st.button("🚀 Analyze Emotion Profile", type="primary", use_container_width=True)

    if analyze_btn or user_text:
        start_time = time.time()

        # 1. Base Predictions
        base_results = engine.predict_base(user_text, model_choice)
        top_classes = base_results if "DistilBERT" in model_choice else base_results[:8]

        labels = [r['label'].capitalize() for r in top_classes]
        scores = [r['score'] for r in top_classes]

        # Information Entropy
        entropy = -sum([r['score'] * math.log(r['score'] + 1e-9) for r in base_results])

        # 2. XAI Occlusion
        top_label, base_score, attributions = engine.get_xai_attribution(user_text, model_choice)

        # 3. Semantic Proximity (200+ Taxonomy)
        nuanced_label, sem_score, _ = engine.predict_semantic(user_text)

        latency_ms = (time.time() - start_time) * 1000

        # Save to history if not duplicate
        if not st.session_state.history or st.session_state.history[-1]['text'] != user_text:
            st.session_state.history.append({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "text": user_text,
                "model": model_choice.split()[0],
                "emotion": top_label.capitalize(),
                "score": f"{base_score*100:.1f}%",
                "nuanced": nuanced_label.capitalize(),
                "entropy": f"{entropy:.2f}",
                "latency": f"{latency_ms:.1f}ms"
            })

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0;'>", unsafe_allow_html=True)

        # ── KPI Cards ────────────────────────────────────────────────────────
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Primary Detected Emotion</div>
                <div class="metric-value" style="color: #60a5fa;">{top_label.upper()}</div>
                <div class="metric-sub">Confidence: <b>{base_score*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Nuanced Semantic Proximity</div>
                <div class="metric-value" style="color: #c084fc;">{nuanced_label.upper()}</div>
                <div class="metric-sub">Cosine Match: <b>{sem_score:.4f}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Uncertainty Entropy</div>
                <div class="metric-value" style="color: #f472b6;">{entropy:.3f} <span style="font-size:12px;">nats</span></div>
                <div class="metric-sub">Information spread index</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Inference Latency</div>
                <div class="metric-value" style="color: #34d399;">{latency_ms:.1f} <span style="font-size:12px;">ms</span></div>
                <div class="metric-sub">Backbone: <b>{model_choice.split()[0]}</b></div>
            </div>
            """, unsafe_allow_html=True)

        # ── Visualizations & XAI Rows ────────────────────────────────────────
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        col_chart, col_xai = st.columns([1.2, 1])

        with col_chart:
            st.subheader("Emotion Confidence Profile")
            if "Donut" in vis_mode:
                pie_colors = [
                    '#6366f1', '#ef4444', '#10b981', '#a855f7', '#f97316',
                    '#06b6d4', '#ec4899', '#84cc16', '#eab308', '#64748b'
                ]
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=scores,
                    hole=0.48,
                    textinfo='label+percent',
                    insidetextorientation='radial',
                    marker=dict(colors=pie_colors[:len(labels)], line=dict(color='#0b0f19', width=2)),
                    textfont=dict(family="Inter", size=12, color="#ffffff")
                )])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0', family="Inter"),
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=380,
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=scores + [scores[0]],
                    theta=labels + [labels[0]],
                    fill='toself',
                    fillcolor='rgba(124, 58, 237, 0.35)',
                    mode='lines+markers',
                    line=dict(color='#a855f7', width=3),
                    marker=dict(size=8, color='#ffffff', line=dict(color='#a855f7', width=2)),
                    name='Emotion Score'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max(0.4, max(scores) * 1.15)], color='#94a3b8', gridcolor='rgba(255,255,255,0.1)'),
                        angularaxis=dict(color='#e2e8f0', gridcolor='rgba(255,255,255,0.1)')
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0', family="Inter"),
                    margin=dict(l=40, r=40, t=20, b=20),
                    height=380,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_xai:
            st.subheader("Occlusion XAI Attribution")
            st.caption(f"Token-level contribution towards predicted **{top_label.upper()}**:")

            xai_chips = []
            for word, score in attributions:
                if score > 0.05:
                    alpha = min(0.85, 0.2 + score * 0.8)
                    bg = f"rgba(16, 185, 129, {alpha:.2f})"
                    border = f"rgba(52, 211, 153, 0.9)"
                    xai_chips.append(f'<span style="display:inline-block; margin:3px 2px; padding:4px 8px; border-radius:6px; background:{bg}; border:1px solid {border}; font-weight:600; color:#ffffff; font-size:13px;" title="Attribution: +{score:.3f}">{word}</span>')
                else:
                    xai_chips.append(f'<span style="display:inline-block; margin:3px 2px; padding:4px 8px; border-radius:6px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); color:#94a3b8; font-size:13px;">{word}</span>')

            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px; line-height: 1.8;">
                {' '.join(xai_chips)}
                <div style="margin-top: 16px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 11px; color: #94a3b8;">
                    🟢 <b>Green tokens:</b> Positive drivers increasing model confidence.<br>
                    ⚪ <b>Grey tokens:</b> Inert context tokens with negligible occlusion impact.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: RESEARCH BENCHMARK MATRIX
# ══════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.subheader("Academic Benchmark Evaluation Matrix")
    st.markdown("Reproduces **Figure 5 and Figure 6** from the research publication across standard emotion datasets.")

    # Convert benchmark data to display table
    bench_data = []
    for model_name, metrics in PAPER_BENCHMARK.items():
        bench_data.append({
            "Backbone Model": model_name,
            "Accuracy": f"{metrics['Accuracy']*100:.1f}%",
            "Macro F1": f"{metrics['Macro F1']*100:.1f}%",
            "Precision": f"{metrics['Precision']*100:.1f}%",
            "Recall": f"{metrics['Recall']*100:.1f}%",
            "Latency (ms)": f"{metrics['Latency']} ms",
            "Taxonomy": metrics.get("Taxonomy", "Standard")
        })

    st.dataframe(bench_data, use_container_width=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.subheader("Live Benchmark Validation Runner")
    col_bench_run, col_bench_res = st.columns([1, 2])

    with col_bench_run:
        eval_model = st.selectbox("Select Model for Live Validation:", ["DistilBERT (Fine-Tuned)", "ModernBERT-Large (GoEmotions)"])
        sample_size = st.slider("Validation Samples:", min_value=10, max_value=50, value=20, step=5)
        run_btn = st.button("⚡ Run Live Validation")

    with col_bench_res:
        if run_btn:
            with st.spinner("Executing real-time inference across test corpus..."):
                metrics, latency_avg = evaluate_live_benchmark(eval_model, sample_size)
                st.success("Validation complete!")
                st.json({
                    "Model": eval_model,
                    "Samples Evaluated": sample_size,
                    "Average Latency": f"{latency_avg:.2f} ms / sample",
                    "Computed Metrics": metrics
                })

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ARCHITECTURE & TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════
with tab_taxonomy:
    st.subheader("Dual Transformer Architecture Ledger")
    st.markdown("""
    ```
    Raw Text Query ────────────────────────┐
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
    [Primary Backbone: ModernBERT / DistilBERT]             [Semantic Embedder: all-MiniLM-L6-v2]
         │                                                                   │
         ▼                                                                   ▼
    Discrete Classification Head (6 or 28 Classes)          768-D Dense Dense Semantic Projection
         │                                                                   │
         ▼                                                                   ▼
    Softmax Probabilities & Information Entropy            Cosine Similarity over 200+ Taxonomy
         └─────────────────────────────────┬─────────────────────────────────┘
                                           ▼
                 Unified Multimodal Emotion Assessment & Occlusion XAI
    ```
    """)

    st.subheader("200+ Emotion Open-Vocabulary Taxonomy")
    search_query = st.text_input("Filter Taxonomy:", placeholder="Search e.g. 'joy', 'melancholy', 'anxiety'...")
    filtered_tax = [e for e in TAXONOMY_200 if search_query.lower() in e.lower()] if search_query else TAXONOMY_200

    st.caption(f"Showing {len(filtered_tax)} out of {len(TAXONOMY_200)} defined emotion states:")
    tax_cols = st.columns(4)
    for idx, emotion in enumerate(filtered_tax):
        tax_cols[idx % 4].markdown(f"• `{emotion}`")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SESSION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("Session Analysis History")
    col_hist_action, _ = st.columns([1, 4])
    with col_hist_action:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    if st.session_state.history:
        st.dataframe(st.session_state.history, use_container_width=True)
    else:
        st.info("No session records yet. Analyze queries in the first tab to build history logs.")

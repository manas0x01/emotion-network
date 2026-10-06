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
from core_engine import (
    EmotionEngine,
    PAPER_BENCHMARK,
    MODEL_BENCHMARKS,
    evaluate_live_benchmark,
    TAXONOMY_200
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmoSphere-XAI | Emotion Intelligence Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Academic Light Theme & Structured CSS System ──────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main application background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Structured Header Banner */
    .formal-hero {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px 0 rgba(0, 0, 0, 0.02);
        border-left: 5px solid #2563eb;
    }
    
    .formal-badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #2563eb;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 3px 8px;
        margin-bottom: 8px;
    }
    
    .formal-title {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    
    .formal-subtitle {
        color: #475569;
        font-size: 13.5px;
        line-height: 1.5;
        margin: 0;
    }

    /* Card Panels */
    .card-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
    }
    
    .card-header-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #334155;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 8px;
    }

    /* Metric Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
        border-top: 3px solid #2563eb;
        height: 100%;
    }
    
    .kpi-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    
    .kpi-sub {
        font-size: 11.5px;
        color: #64748b;
        margin-top: 6px;
    }

    /* Structured Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 13px;
        color: #475569;
        border: 1px solid transparent;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #1e293b;
        background: #f1f5f9;
    }
    
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        background: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
    }

    /* Input text areas and select boxes */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 1px #2563eb !important;
    }

    /* Secondary preset buttons */
    div[data-testid="column"] button[kind="secondary"] {
        border: 1px solid #cbd5e1 !important;
        background-color: #ffffff !important;
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 12px !important;
        border-radius: 6px !important;
        transition: all 0.15s ease-in-out;
    }
    
    div[data-testid="column"] button[kind="secondary"]:hover {
        border-color: #94a3b8 !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Primary buttons */
    button[kind="primary"] {
        background-color: #2563eb !important;
        border: 1px solid #1d4ed8 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px 0 rgba(37, 99, 235, 0.2) !important;
    }

    /* Token Attribution Chips */
    .token-chip-pos {
        display: inline-block;
        margin: 3px 2px;
        padding: 4px 8px;
        border-radius: 6px;
        background: #dcfce7;
        border: 1px solid #86efac;
        color: #166534;
        font-weight: 600;
        font-size: 13px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .token-chip-neutral {
        display: inline-block;
        margin: 3px 2px;
        padding: 4px 8px;
        border-radius: 6px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #475569;
        font-size: 13px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Clean spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
</style>
""", unsafe_allow_html=True)

# ── Engine Singleton via st.cache_resource ───────────────────────────────────
@st.cache_resource(show_spinner="Initializing Neural Models & Semantic Embeddings...")
def get_engine():
    return EmotionEngine()

# ── Session State Initializers ────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = "I feel an overwhelming sense of joy, relief and gratitude that everything finally worked out!"

if "has_run" not in st.session_state:
    st.session_state.has_run = False

# ── Structured Institutional Header ──────────────────────────────────────────
st.markdown("""
<div class="formal-hero">
    <div class="formal-badge">RESEARCH FRAMEWORK · DUAL TRANSFORMER ARCHITECTURE</div>
    <h1 class="formal-title">EmoSphere-XAI: Emotion Intelligence Platform</h1>
    <p class="formal-subtitle">
        Fine-Tuned DistilBERT & ModernBERT-Large Backbone &bull; 200+ Open-Vocabulary Concept Mapping &bull; Real-Time Occlusion Explainability (XAI)
    </p>
</div>
""", unsafe_allow_html=True)

# ── Formal Navigation Tabs ───────────────────────────────────────────────────
tab_sandbox, tab_benchmark, tab_taxonomy, tab_history = st.tabs([
    "◈ Diagnostic Inference & XAI",
    "⊞ Benchmark Evaluation Matrix",
    "☵ Architecture & Taxonomy",
    "◷ Session Audit Log"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DIAGNOSTIC INFERENCE & XAI
# ══════════════════════════════════════════════════════════════════════════════
with tab_sandbox:
    col_input, col_config = st.columns([2.3, 1], gap="medium")

    with col_input:
        st.markdown("""
        <div class="card-header-title">
            <span>[1] Input Text Corpus for Emotion Analysis</span>
        </div>
        """, unsafe_allow_html=True)
        
        user_text = st.text_area(
            "Input Text:",
            value=st.session_state.prompt_text,
            height=120,
            placeholder="Enter clinical, conversational, or textual content to evaluate emotional distribution...",
            label_visibility="collapsed"
        )

        st.caption("Standardized Academic Test Prompts:")
        c1, c2, c3 = st.columns(3)
        if c1.button("Sample 1: Joy & Relief", use_container_width=True):
            st.session_state.prompt_text = "I feel an overwhelming sense of joy, relief and gratitude that everything finally worked out!"
            st.session_state.has_run = True
            st.rerun()
        if c2.button("Sample 2: Grief & Betrayal", use_container_width=True):
            st.session_state.prompt_text = "I am utterly devastated and betrayed by the broken promises."
            st.session_state.has_run = True
            st.rerun()
        if c3.button("Sample 3: Dread & Anxiety", use_container_width=True):
            st.session_state.prompt_text = "The dark silence in the empty hallway filled me with creeping dread and anxiety."
            st.session_state.has_run = True
            st.rerun()

    with col_config:
        st.markdown("""
        <div class="card-header-title">
            <span>[2] Model Configuration</span>
        </div>
        """, unsafe_allow_html=True)

        model_choice = st.selectbox(
            "Transformer Backbone:",
            [
                "ModernBERT-Large (GoEmotions 28-Class)",
                "DistilBERT (Fine-Tuned 6-Class)",
            ],
            index=0
        )

        vis_mode = st.radio(
            "Visualization Projection:",
            ["Donut Distribution", "Polar Radar Profile"],
            horizontal=True
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("Execute Diagnostic Inference", type="primary", use_container_width=True)

    if analyze_btn:
        st.session_state.has_run = True
        st.session_state.prompt_text = user_text

    if st.session_state.has_run and user_text.strip():
        with st.spinner("Executing neural inference & occlusion attribution..."):
            engine = get_engine()
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
                    "entropy": f"{entropy:.3f}",
                    "latency": f"{latency_ms:.1f}ms"
                })

        st.markdown("<hr style='border-color: #e2e8f0; margin: 24px 0;'>", unsafe_allow_html=True)

        # ── KPI Cards ────────────────────────────────────────────────────────
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #2563eb;">
                <div class="kpi-title">Primary Classification</div>
                <div class="kpi-value" style="color: #1d4ed8;">{top_label.upper()}</div>
                <div class="kpi-sub">Confidence: <b>{base_score*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #7c3aed;">
                <div class="kpi-title">Semantic Proximity Concept</div>
                <div class="kpi-value" style="color: #6d28d9;">{nuanced_label.upper()}</div>
                <div class="kpi-sub">Cosine Match: <b>{sem_score:.4f}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #d97706;">
                <div class="kpi-title">Shannon Uncertainty Entropy</div>
                <div class="kpi-value" style="color: #b45309;">{entropy:.3f} <span style="font-size:12px; font-weight:400; color:#64748b;">nats</span></div>
                <div class="kpi-sub">Prediction confidence spread</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #059669;">
                <div class="kpi-title">Inference Execution Latency</div>
                <div class="kpi-value" style="color: #047857;">{latency_ms:.1f} <span style="font-size:12px; font-weight:400; color:#64748b;">ms</span></div>
                <div class="kpi-sub">Backbone: <b>{model_choice.split()[0]}</b></div>
            </div>
            """, unsafe_allow_html=True)

        # ── Visualizations & XAI Rows ────────────────────────────────────────
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        col_chart, col_xai = st.columns([1.2, 1], gap="medium")

        with col_chart:
            st.markdown("""
            <div class="card-header-title">
                <span>Confidence Distribution Matrix</span>
            </div>
            """, unsafe_allow_html=True)
            
            if "Donut" in vis_mode:
                formal_colors = [
                    '#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626',
                    '#0891b2', '#4f46e5', '#ca8a04', '#64748b', '#0284c7'
                ]
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=scores,
                    hole=0.52,
                    textinfo='label+percent',
                    insidetextorientation='radial',
                    marker=dict(colors=formal_colors[:len(labels)], line=dict(color='#ffffff', width=2)),
                    textfont=dict(family="Inter", size=12, color="#0f172a")
                )])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#334155', family="Inter"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=360,
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=scores + [scores[0]],
                    theta=labels + [labels[0]],
                    fill='toself',
                    fillcolor='rgba(37, 99, 235, 0.12)',
                    mode='lines+markers',
                    line=dict(color='#2563eb', width=2.5),
                    marker=dict(size=7, color='#1d4ed8'),
                    name='Confidence Score'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max(0.4, max(scores) * 1.15)], color='#64748b', gridcolor='#e2e8f0'),
                        angularaxis=dict(color='#334155', gridcolor='#e2e8f0')
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#334155', family="Inter"),
                    margin=dict(l=30, r=30, t=10, b=10),
                    height=360,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_xai:
            st.markdown("""
            <div class="card-header-title">
                <span>Occlusion-Based Token Attribution (XAI)</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"Differential impact on target class **{top_label.upper()}** upon token masking:")

            xai_chips = []
            for word, score in attributions:
                if score > 0.05:
                    xai_chips.append(f'<span class="token-chip-pos" title="Occlusion Impact: +{score:.3f}">{word} <small style="font-size:10px; opacity:0.8;">(+{score:.2f})</small></span>')
                else:
                    xai_chips.append(f'<span class="token-chip-neutral">{word}</span>')

            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; line-height: 1.9; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.02);">
                {' '.join(xai_chips)}
                <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #f1f5f9; font-size: 11.5px; color: #64748b;">
                    <span style="color:#166534; font-weight:700;">■ Green Highlight:</span> Positive attribution token driving model prediction.<br>
                    <span style="color:#64748b; font-weight:700;">■ Grey Outline:</span> Low-impact contextual token with negligible occlusion delta.
                </div>
            </div>
            """, unsafe_allow_html=True)
    elif not st.session_state.has_run:
        st.markdown("<hr style='border-color: #e2e8f0; margin: 24px 0;'>", unsafe_allow_html=True)
        st.info("Select a standardized sample prompt above or enter custom text, then click **Execute Diagnostic Inference** to analyze the emotional distribution and generate the occlusion token attribution map.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: RESEARCH BENCHMARK MATRIX
# ══════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.markdown("""
    <div class="card-header-title" style="margin-top: 8px;">
        <span>Figure 5: Overall Research Model Performance (dair-ai/emotion Test Corpus)</span>
    </div>
    """, unsafe_allow_html=True)

    col_kpi_a, col_kpi_b, col_kpi_c, col_kpi_d = st.columns(4)
    with col_kpi_a:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #059669;">
            <div class="kpi-title">Accuracy Rate</div>
            <div class="kpi-value" style="color: #047857;">{PAPER_BENCHMARK['accuracy']*100:.1f}%</div>
            <div class="kpi-sub">1,722 / 2,000 Verified Test Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi_b:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #2563eb;">
            <div class="kpi-title">Precision Index</div>
            <div class="kpi-value" style="color: #1d4ed8;">{PAPER_BENCHMARK['precision']:.4f}</div>
            <div class="kpi-sub">Weighted precision across classes</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi_c:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #7c3aed;">
            <div class="kpi-title">Recall Index</div>
            <div class="kpi-value" style="color: #6d28d9;">{PAPER_BENCHMARK['recall']:.4f}</div>
            <div class="kpi-sub">Macro sensitivity rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi_d:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #d97706;">
            <div class="kpi-title">Macro F1-Score</div>
            <div class="kpi-value" style="color: #b45309;">{PAPER_BENCHMARK['f1_score']:.4f}</div>
            <div class="kpi-sub">Harmonic mean balance</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card-header-title">
        <span>Table I: Comparative Architecture Evaluation Matrix</span>
    </div>
    """, unsafe_allow_html=True)

    bench_data = []
    for model_name, metrics in MODEL_BENCHMARKS.items():
        bench_data.append({
            "Backbone Architecture": model_name,
            "Accuracy": f"{metrics['Accuracy']*100:.1f}%",
            "Macro F1": f"{metrics['Macro F1']*100:.1f}%",
            "Precision": f"{metrics['Precision']*100:.1f}%",
            "Recall": f"{metrics['Recall']*100:.1f}%",
            "Mean Latency": f"{metrics['Latency']:.1f} ms",
            "Target Taxonomy": metrics.get("Taxonomy", "Standard")
        })

    st.dataframe(bench_data, use_container_width=True)

    st.markdown("<hr style='border-color: #e2e8f0; margin: 24px 0;'>", unsafe_allow_html=True)

    # ── Figure 6: Confusion Matrix Heatmap ────────────────────────────────────
    st.markdown("""
    <div class="card-header-title">
        <span>Figure 6: Confusion Matrix Across Emotion Categories (DistilBERT N=2,000 Test Set)</span>
    </div>
    """, unsafe_allow_html=True)

    cm = PAPER_BENCHMARK["confusion_matrix"]
    labels = PAPER_BENCHMARK["labels"]

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#f8fafc"],
            [0.1, "#dbeafe"],
            [0.4, "#93c5fd"],
            [0.7, "#3b82f6"],
            [1.0, "#1d4ed8"]
        ],
        text=[[str(val) for val in row] for row in cm],
        texttemplate="<b>%{text}</b>",
        textfont={"size": 13, "color": "#0f172a"},
        hoverongaps=False,
        colorbar=dict(title="Sample Count")
    ))
    fig_cm.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#334155', family="Inter"),
        xaxis_title="Predicted Class",
        yaxis_title="True Ground Truth",
        height=380,
        margin=dict(l=40, r=40, t=10, b=40)
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<hr style='border-color: #e2e8f0; margin: 24px 0;'>", unsafe_allow_html=True)

    # ── Live Benchmark Validation Runner ─────────────────────────────────────
    st.markdown("""
    <div class="card-header-title">
        <span>Corpus Validation Protocol</span>
    </div>
    """, unsafe_allow_html=True)

    col_bench_run, col_bench_res = st.columns([1, 2], gap="medium")

    with col_bench_run:
        eval_model = st.selectbox("Validation Model Target:", ["DistilBERT (Fine-Tuned)", "ModernBERT-Large (GoEmotions)"])
        sample_size = st.slider("Evaluation Sample Set:", min_value=10, max_value=50, value=20, step=5)
        run_btn = st.button("Execute Corpus Validation", type="primary", use_container_width=True)

    with col_bench_res:
        if run_btn:
            with st.spinner("Executing real-time inference across test corpus..."):
                metrics, latency_avg = evaluate_live_benchmark(eval_model, sample_size)
                st.success("Corpus validation finished successfully.")
                st.json({
                    "Target Architecture": eval_model,
                    "Evaluated Samples": sample_size,
                    "Empirical Mean Latency": f"{latency_avg:.2f} ms / sample",
                    "Validated Performance Metrics": metrics
                })

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ARCHITECTURE & TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════
with tab_taxonomy:
    st.markdown("""
    <div class="card-header-title" style="margin-top: 8px;">
        <span>System Architecture & Information Flow Diagram</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ```
    Input Query Text
           │
           ├───────────────────────────────────────────────┐
           ▼                                               ▼
    [Primary Backbone: ModernBERT / DistilBERT]    [Semantic Encoder: all-MiniLM-L6-v2]
           │                                               │
           ▼                                               ▼
    Discrete Classification (6 / 28 Classes)       768-D Dense Semantic Projection
           │                                               │
           ▼                                               ▼
    Softmax Distribution & Shannon Entropy         Cosine Similarity against 200+ Lexicon
           └───────────────────────┬───────────────────────┘
                                   ▼
          Unified Multimodal Emotion Assessment & Occlusion XAI
    ```
    """)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card-header-title">
        <span>Open-Vocabulary Affective Concept Taxonomy (200+ Entries)</span>
    </div>
    """, unsafe_allow_html=True)

    search_query = st.text_input("Filter Concept Lexicon:", placeholder="Type emotion concept e.g., 'melancholy', 'serenity', 'rage'...")
    filtered_tax = [e for e in TAXONOMY_200 if search_query.lower() in e.lower()] if search_query else TAXONOMY_200

    st.caption(f"Displaying {len(filtered_tax)} out of {len(TAXONOMY_200)} defined emotion states:")
    tax_cols = st.columns(4)
    for idx, emotion in enumerate(filtered_tax):
        tax_cols[idx % 4].markdown(f"• `{emotion}`")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SESSION AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("""
    <div class="card-header-title" style="margin-top: 8px;">
        <span>Session Inference Audit Log</span>
    </div>
    """, unsafe_allow_html=True)

    col_hist_action, _ = st.columns([1, 4])
    with col_hist_action:
        if st.button("Clear Audit Log", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    if st.session_state.history:
        st.dataframe(st.session_state.history, use_container_width=True)
    else:
        st.info("No recorded inference cycles in current session. Execute analysis in the diagnostic tab to register audit entries.")

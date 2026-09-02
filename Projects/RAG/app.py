import os
import time
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG Core (Preserved from prior application)
from core.document_loader import load_document_from_bytes
from core.chunking import chunk_documents
from core.vector_store import VectorStore
from core.llm import OpenRouterClient, RAGPipeline, POPULAR_OPENROUTER_MODELS, DEFAULT_SYSTEM_PROMPT

# Import AI PM Exploration Toolkit Modules
from pm_toolkit.llm_engine import LLMEngine
from pm_toolkit.pol_probes import PoLProbeStudio
from pm_toolkit.synthetic_personas import SyntheticPersonaGenerator
from pm_toolkit.audio_intelligence import AudioIntelligenceLab
from pm_toolkit.competitive_intel import CompetitiveIntelligenceLab
from pm_toolkit.prompt_evaluator import PromptEvaluator
from pm_toolkit.executive_narrative import ExecutiveNarrativeCrafter
from pm_toolkit.learning_classroom import LearningClassroom

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Abhinav's AI PM Exploration Studio | OpenRouter",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: white;
        padding: 24px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .main-header h1 {
        color: #F8FAFC;
        font-size: 2.1rem;
        margin: 0 0 6px 0;
        font-weight: 700;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 1.05rem;
        margin: 0;
    }
    .user-tag {
        background: rgba(99, 102, 241, 0.25);
        color: #A5B4FC;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    .pillar-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: transform 0.15s ease-in-out;
    }
    .pillar-card:hover {
        border-color: #CBD5E1;
        background: #FFFFFF;
    }
    .pillar-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .badge-urgent { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
    .badge-schedule { background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
    .badge-low { background-color: #E0E7FF; color: #3730A3; border: 1px solid #C7D2FE; }
    .quote-box {
        border-left: 4px solid #6366F1;
        background: #F1F5F9;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-style: italic;
        color: #334155;
    }
    .metric-pill {
        background: #EEF2F6;
        padding: 6px 12px;
        border-radius: 6px;
        display: inline-block;
        font-weight: 600;
        margin-right: 8px;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# State Initialization
# ==============================================================================
if "llm_engine" not in st.session_state:
    st.session_state.llm_engine = LLMEngine()

if "pol_studio" not in st.session_state:
    st.session_state.pol_studio = PoLProbeStudio()

if "persona_gen" not in st.session_state:
    st.session_state.persona_gen = SyntheticPersonaGenerator()

if "audio_lab" not in st.session_state:
    st.session_state.audio_lab = AudioIntelligenceLab()

if "comp_lab" not in st.session_state:
    st.session_state.comp_lab = CompetitiveIntelligenceLab()

if "prompt_eval" not in st.session_state:
    st.session_state.prompt_eval = PromptEvaluator()

if "narrative_crafter" not in st.session_state:
    st.session_state.narrative_crafter = ExecutiveNarrativeCrafter()

if "classroom" not in st.session_state:
    st.session_state.classroom = LearningClassroom()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = [
        {"role": "assistant", "content": "👋 **Welcome, Abhinav!** Upload documents to search and query them with transparent citations powered by OpenRouter."}
    ]

if "copilot_history" not in st.session_state:
    st.session_state.copilot_history = [
        {
            "role": "assistant",
            "content": "👋 **Hello Abhinav! I'm your AI PM Strategic Co-Pilot.**\n\nI operate under Dean Peters' **Proof-of-Life (PoL) Probes** framework. Ask me to evaluate feature ideas, design spike-and-delete tests, craft PRDs, or diagnose competitive moats."
        }
    ]

# ==============================================================================
# SIDEBAR CONTROLS & ENVIRONMENT
# ==============================================================================
with st.sidebar:
    st.markdown("### 👤 Product Lead")
    st.markdown(
        """
        <div style="background: #F1F5F9; border: 1px solid #CBD5E1; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
            <div style="font-weight: 700; color: #0F172A; font-size: 1.05rem;">Abhinav</div>
            <div style="color: #64748B; font-size: 0.85rem;">Lead Product Manager | AI Exploration</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 🧭 Application Mode")
    app_mode = st.radio(
        "Select Workspace Mode:",
        ["🚀 AI PM Exploration Studio", "📚 AI Document RAG Assistant"],
        index=0,
        help="Switch between Abhinav's AI PM Exploration Studio and the Document RAG Assistant."
    )
    
    st.divider()
    st.markdown("### 🌐 OpenRouter Configuration")
    
    # OpenRouter API Key Input
    api_key_env = os.getenv("OPENROUTER_API_KEY", "")
    api_key_input = st.text_input(
        "OpenRouter API Key:",
        value=api_key_env,
        type="password",
        placeholder="sk-or-v1-...",
        help="Enter your OpenRouter API key from openrouter.ai/keys (or set OPENROUTER_API_KEY in .env)"
    )

    # OpenRouter Model Selection
    popular_openrouter_models = [
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "anthropic/claude-3.5-sonnet",
        "deepseek/deepseek-r1",
        "meta-llama/llama-3.3-70b-instruct",
        "mistralai/mistral-large"
    ]
    
    selected_openrouter_model = st.selectbox(
        "OpenRouter Model:",
        popular_openrouter_models,
        index=0,
        help="Choose state-of-the-art models for real-time analysis."
    )
    
    # Detect Providers
    providers = st.session_state.llm_engine.detect_providers(openrouter_key=api_key_input)
    
    provider_options = []
    if api_key_input and len(api_key_input.strip()) > 5:
        provider_options.append("OpenRouter Cloud API (Recommended)")
    if providers["ollama"]["available"]:
        provider_options.append("Local Ollama")
    provider_options.append("Offline PM Simulation Engine")
    
    selected_provider_label = st.selectbox(
        "Active AI Provider:",
        provider_options,
        index=0
    )
    
    active_provider = "auto"
    if "OpenRouter" in selected_provider_label:
        active_provider = "openrouter"
    elif "Ollama" in selected_provider_label:
        active_provider = "ollama"
    else:
        active_provider = "offline"

    # Status Pill
    if api_key_input and len(api_key_input.strip()) > 5:
        st.success(f"🟢 OpenRouter: Connected ({selected_openrouter_model.split('/')[-1]})")
    else:
        st.info("ℹ️ OpenRouter: No key detected (Using Offline Simulation)")
        st.caption("Get a key at [openrouter.ai/keys](https://openrouter.ai/keys) or add to `.env`")
        
    if providers["ollama"]["available"]:
        st.success("🟢 Local Ollama: Online")
        
    st.divider()
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #64748B;">
        <b>Abhinav's AI PM Toolkit</b><br>
        <i>Powered by OpenRouter & PoL Probes</i><br>
        "Use the cheapest prototype that tells the harshest truth."
        </div>
        """,
        unsafe_allow_html=True
    )



# ==============================================================================
# MODE 1: AI PM EXPLORATION STUDIO (4E FRAMEWORK)
# ==============================================================================
if app_mode == "🚀 AI PM Exploration Studio":
    # Header Banner
    st.markdown(
        """
        <div class="main-header">
            <div class="user-tag">👤 Lead Product Manager: Abhinav</div>
            <h1>🚀 Abhinav's AI PM Exploration Studio</h1>
            <p>Welcome, <b>Abhinav</b>! A safe, hands-on learning laboratory for Product Managers to turn FOMO into fluency using the <b>4E Framework</b> & <b>Proof-of-Life (PoL) Probes</b> powered by OpenRouter.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Main Navigation Tabs for the 4E Journey
    tab_classroom, tab_pol, tab_personas, tab_audio, tab_competitor, tab_prompts, tab_narrative, tab_copilot = st.tabs([
        "🎓 1. PM Classroom",
        "🧪 2. PoL Probe Studio",
        "👥 3. Synthetic Personas",
        "🎙️ 4. Audio Customer Voice",
        "🛡️ 5. Competitive Moat",
        "🎛️ 6. Prompt Sandbox",
        "📄 7. Executive PRD Crafter",
        "💬 8. Strategic Co-Pilot"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: PM CLASSROOM (EDUCATION)
    # --------------------------------------------------------------------------
    with tab_classroom:
        st.markdown("## 🎓 The Personal AI Classroom")
        st.markdown("Close the AI skills gap through structured mental models, evaluation frameworks, and hands-on fluency.")
        
        # 4E Framework Cards
        f4e = st.session_state.classroom.get_4e_framework_breakdown()
        st.markdown(f"### {f4e['title']}")
        cols = st.columns(4)
        for idx, pillar in enumerate(f4e["pillars"]):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="pillar-card">
                        <div style="font-size: 1.8rem;">{pillar['emoji']}</div>
                        <div class="pillar-title">{pillar['name']}</div>
                        <p style="font-size: 0.88rem; color: #475569;">{pillar['description']}</p>
                        <hr style="margin: 8px 0; border: none; border-top: 1px solid #E2E8F0;">
                        <small style="color: #6366F1; font-weight: 600;">Action: {pillar['action']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.divider()
        st.markdown("### 🧠 Core PM AI Mental Models")
        models = st.session_state.classroom.get_mental_models()
        for m in models:
            with st.expander(f"📌 {m['title']}", expanded=True if "Probabilistic" in m["title"] else False):
                st.markdown(m["summary"])

        st.divider()
        st.markdown("### 📝 PM AI Fluency Self-Assessment Quiz")
        questions = st.session_state.classroom.get_fluency_quiz_questions()
        for q_idx, q in enumerate(questions):
            st.markdown(f"**Q{q_idx+1}: {q['question']}**")
            user_choice = st.radio(
                f"Options for Q{q_idx+1}:",
                q["options"],
                key=f"quiz_q_{q_idx}",
                label_visibility="collapsed"
            )
            choice_idx = q["options"].index(user_choice)
            if choice_idx == q["correct_idx"]:
                st.success(f"✅ {q['explanation']}")
            else:
                st.info("💡 Review the PoL Probe principles above to see why option 2 is recommended.")
            st.write("")

    # --------------------------------------------------------------------------
    # TAB 2: PROOF-OF-LIFE (PoL) PROBE STUDIO (EXPERIMENTATION)
    # --------------------------------------------------------------------------
    with tab_pol:
        st.markdown("## 🧪 Proof-of-Life (PoL) Probe Studio")
        st.markdown(
            """
            > *"The most expensive way to test your idea is to build production-quality software."* ~ **Jeff Patton**  
            > **PoL Probes** are disposable, 1–2 day reconnaissance tests to kill bad ideas before sprint planning.
            """
        )

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### 1. Configure Your Feature Probe")
            preset_key = st.selectbox(
                "Load a Real-World PM Scenario (or custom):",
                ["Custom", "ai_copilot", "smart_search", "automated_summary"],
                format_func=lambda x: "Custom Scenario" if x == "Custom" else st.session_state.pol_studio.preset_scenarios[x]["name"]
            )

            if preset_key != "Custom":
                preset_data = st.session_state.pol_studio.preset_scenarios[preset_key]
                default_name = preset_data["name"]
                default_hypo = preset_data["hypothesis"]
                default_risk = preset_data["risk_area"]
                default_flavor = preset_data["cheapest_probe"]
                default_kill = preset_data["kill_criteria"]
            else:
                default_name = "AI Weekly Status Summarizer"
                default_hypo = "Providing automated weekly status briefs will reduce sprint planning friction by 40%."
                default_risk = "Engineers and PMs spending time manually verifying summaries"
                default_flavor = "feasibility_check"
                default_kill = "If users spend > 5 mins correcting the AI summary, kill the feature."

            probe_feature = st.text_input("Feature / Idea Name:", value=default_name)
            probe_hypothesis = st.text_area("Core Hypothesis (We believe that...):", value=default_hypo, height=70)
            probe_flavor = st.selectbox(
                "Select PoL Probe Flavor:",
                list(st.session_state.pol_studio.probe_flavors.keys()),
                index=list(st.session_state.pol_studio.probe_flavors.keys()).index(default_flavor) if default_flavor in st.session_state.pol_studio.probe_flavors else 0,
                format_func=lambda k: st.session_state.pol_studio.probe_flavors[k].name
            )
            probe_lethal = st.text_input("The Lethal Assumption (What kills this if wrong?):", value=default_risk)
            probe_pass = st.text_input("Pass Criteria (When to proceed):", value=">= 70% of test users rate summary as production-ready without edits.")
            probe_kill = st.text_input("Kill Criteria (When to pull the plug):", value=default_kill)
            probe_days = st.slider("Execution Window (Days):", 1, 5, 2)

        with col_right:
            st.markdown("### 2. Probe Priority & Urgency Matrix")
            st.markdown("Calculate whether this probe should be prioritized today vs backlogged.")
            
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                imp = st.slider("Potential Value/Impact (1-10):", 1, 10, 8)
                unc = st.slider("Uncertainty / Risk Unknown (1-10):", 1, 10, 9)
            with c_m2:
                eff = st.slider("Probe Build Cost/Effort (1-10):", 1, 10, 2, help="Lower score means cheaper to test")
                aln = st.slider("OKR / Strategic Alignment (1-10):", 1, 10, 8)

            score_data = st.session_state.pol_studio.calculate_priority_score(imp, unc, eff, aln)
            
            badge_class = "badge-urgent" if score_data["score"] >= 75 else ("badge-schedule" if score_data["score"] >= 50 else "badge-low")
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; margin-top: 10px;">
                    <div style="font-size: 0.9rem; color: #64748B;">CALCULATED PROBE PRIORITY INDEX</div>
                    <div style="font-size: 2.4rem; font-weight: 800; color: #1E293B;">{score_data['score']} / 100</div>
                    <span class="score-badge {badge_class}">{score_data['badge']}</span>
                    <div style="margin-top: 12px; font-weight: 600; color: #334155;">{score_data['recommendation']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()
        st.markdown("### 3. Generated PoL Probe Test Card")
        test_card_md = st.session_state.pol_studio.generate_test_card(
            feature_name=probe_feature,
            hypothesis=probe_hypothesis,
            probe_flavor=probe_flavor,
            lethal_assumption=probe_lethal,
            success_criteria=probe_pass,
            kill_criteria=probe_kill,
            duration_days=probe_days,
            lead_pm="Abhinav"
        )
        st.markdown(test_card_md)
        st.download_button(
            "📥 Download PoL Test Card (Markdown)",
            data=test_card_md,
            file_name=f"pol_test_card_{probe_flavor}.md",
            mime="text/markdown"
        )

    # --------------------------------------------------------------------------
    # TAB 3: SYNTHETIC PERSONAS & WIND TUNNEL (EXPERIMENTATION)
    # --------------------------------------------------------------------------
    with tab_personas:
        st.markdown("## 👥 Synthetic Persona & Market 'Wind Tunnel' Simulator")
        st.markdown("Stress-test your feature ideas against realistic synthetic customer cohorts before burning customer relationship capital.")

        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            st.markdown("#### 1. Generate Persona Cohort")
            chosen_ind = st.selectbox(
                "Target Industry:",
                ["b2b_saas", "b2c_consumer", "fintech", "healthtech", "ai_agents"],
                format_func=lambda x: x.replace("_", " ").upper()
            )
            p_count = st.slider("Number of Personas:", 2, 8, 4)
            
            if st.button("🔄 Generate Synthetic Cohort", use_container_width=True):
                st.session_state.active_personas = st.session_state.persona_gen.generate_personas(chosen_ind, count=p_count)
                st.success(f"Generated {p_count} synthetic {chosen_ind.upper()} personas!")

        if "active_personas" not in st.session_state:
            st.session_state.active_personas = st.session_state.persona_gen.generate_personas("b2b_saas", count=4)

        with p_col2:
            st.markdown("#### 2. Generated Persona Cohort Overview")
            df_personas = st.session_state.persona_gen.personas_to_dataframe(st.session_state.active_personas)
            st.dataframe(df_personas, use_container_width=True)

        st.divider()
        st.markdown("### 🌪️ Run Feature 'Wind Tunnel' Simulation")
        st.markdown("Input a proposed feature concept to simulate how each persona in this cohort reacts.")
        
        feature_test_input = st.text_input(
            "Proposed Feature Concept:",
            value="Automate weekly sprint status briefs with an AI copilot directly from GitHub commits",
            placeholder="Describe the feature or workflow to test..."
        )
        
        if st.button("🚀 Run Wind Tunnel Test", type="primary"):
            with st.spinner("Simulating persona reactions and willingness-to-pay..."):
                time.sleep(0.4)
                sim_res = st.session_state.persona_gen.run_wind_tunnel_simulation(
                    feature_test_input,
                    st.session_state.active_personas
                )
                
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Cohort Simulated", sim_res["total_simulated"])
            with m2:
                st.metric("Acceptance Rate", f"{sim_res['acceptance_rate_pct']}%")
            with m3:
                st.metric("Enthusiastic", sim_res["sentiment_summary"]["Enthusiastic"])
            with m4:
                st.metric("Skeptical / Opposed", sim_res["sentiment_summary"]["Skeptical"] + sim_res["sentiment_summary"]["Opposed"])

            st.markdown("#### Simulated Customer Reactions & Quotes")
            for r in sim_res["detailed_reactions"]:
                sent_emoji = "🟢" if r["sentiment"] == "Enthusiastic" else ("🟡" if r["sentiment"] == "Skeptical" else "🔴")
                with st.expander(f"{sent_emoji} {r['name']} ({r['title']} @ {r['company']}) — Sentiment: {r['sentiment']}"):
                    st.markdown(f"**Verbatim Reaction:**")
                    st.markdown(f"> *\"{r['simulated_quote']}\"*")
                    st.markdown(f"**Primary Objection Raised:** `{r['primary_objection']}`")
                    st.markdown(f"**Estimated Willingness-to-Pay:** `${r['estimated_wtp_usd']}`")

            # CSV Download
            csv_data = df_personas.to_csv(index=False)
            st.download_button(
                "📥 Export Personas & Simulation (CSV)",
                data=csv_data,
                file_name=f"synthetic_cohort_{chosen_ind}.csv",
                mime="text/csv"
            )

    # --------------------------------------------------------------------------
    # TAB 4: AUDIO CUSTOMER VOICE LAB (EXPLORATION)
    # --------------------------------------------------------------------------
    with tab_audio:
        st.markdown("## 🎙️ Audio Intelligence & Customer Voice Lab")
        st.markdown("Phase 7.1 Dean Peters PoL Framework: 6 PM-Specific Audio Workflows to extract actionable product signal.")

        a_col1, a_col2 = st.columns([1, 1])
        with a_col1:
            st.markdown("#### 1. Select Workflow Template")
            selected_tmpl_id = st.selectbox(
                "PM Workflow Template:",
                list(st.session_state.audio_lab.templates.keys()),
                format_func=lambda k: st.session_state.audio_lab.templates[k].name
            )
            tmpl_info = st.session_state.audio_lab.templates[selected_tmpl_id]
            st.info(f"**Output Goal:** {tmpl_info.target_output}\n\n**Focus:** {', '.join(tmpl_info.pm_focus_areas)}")

            audio_file = st.file_uploader("Upload Audio Recording (.mp3, .wav, .m4a):", type=["mp3", "wav", "m4a"])
            if audio_file:
                st.audio(audio_file)
                st.success(f"Attached audio: {audio_file.name} ({round(audio_file.size/1024, 1)} KB)")

        with a_col2:
            st.markdown("#### 2. Transcript Input / Sample Presets")
            sample_key = selected_tmpl_id if selected_tmpl_id in st.session_state.audio_lab.sample_transcripts else "user_interview"
            use_sample = st.checkbox("Load realistic pre-recorded transcript sample", value=True)
            
            transcript_input = st.text_area(
                "Transcript Content:",
                value=st.session_state.audio_lab.sample_transcripts.get(sample_key, "") if use_sample else "",
                height=180,
                placeholder="Paste interview transcript or speech-to-text output here..."
            )

        if st.button("⚡ Process PM Audio Intelligence Workflow", type="primary", use_container_width=True):
            if not transcript_input.strip():
                st.warning("Please provide a transcript or load a sample preset.")
            else:
                with st.spinner("Extracting pain points, decisions, quotes, and PoL probe recommendations..."):
                    audio_res = st.session_state.audio_lab.process_transcript(selected_tmpl_id, transcript_input)
                
                st.divider()
                st.markdown(f"### 📋 Extracted PM Intelligence: {audio_res['template_name']}")
                
                res_col1, res_col2 = st.columns(2)
                focus_keys = list(audio_res["extracted_sections"].keys())
                half = (len(focus_keys) + 1) // 2
                
                with res_col1:
                    for k in focus_keys[:half]:
                        with st.container():
                            st.markdown(f"**🔹 {k}**")
                            for item in audio_res["extracted_sections"][k]:
                                st.markdown(f"- {item}")
                            st.write("")
                            
                with res_col2:
                    for k in focus_keys[half:]:
                        with st.container():
                            st.markdown(f"**🔹 {k}**")
                            for item in audio_res["extracted_sections"][k]:
                                st.markdown(f"- {item}")
                            st.write("")

    # --------------------------------------------------------------------------
    # TAB 5: COMPETITIVE MOAT & INTELLIGENCE (EXPLORATION)
    # --------------------------------------------------------------------------
    with tab_competitor:
        st.markdown("## 🛡️ Competitive Intelligence & Moat Analyzer")
        st.markdown("Evaluate competitor vulnerabilities, Hamilton Helmer 7 Powers Moats, and counter-positioning strategies.")

        c_col1, c_col2 = st.columns([1, 1])
        with c_col1:
            comp_preset = st.selectbox(
                "Select Competitor Archetype:",
                list(st.session_state.comp_lab.preset_competitors.keys())
            )
            archetype = st.session_state.comp_lab.preset_competitors[comp_preset]
            our_product = st.text_input("Your Product Name:", value="Abhinav's AI PM Studio")
            competitor_name = st.text_input("Competitor Name:", value=archetype.name)
            market_sector = st.text_input("Market Sector:", value=archetype.category)

        with c_col2:
            st.markdown("#### Archetype Breakdown")
            st.markdown(f"**Target Segment:** {archetype.target_segment}")
            st.markdown(f"**Core Strength:** {archetype.core_strength}")
            st.markdown(f"**Glaring Weakness:** {archetype.glaring_weakness}")
            st.markdown(f"**Pricing Model:** {archetype.pricing_model}")

        if st.button("📊 Run Moat & Vulnerability Analysis", type="primary"):
            moat_data = st.session_state.comp_lab.analyze_moat_and_vulnerabilities(our_product, competitor_name, market_sector)
            
            st.divider()
            st.markdown("### 🏰 Hamilton Helmer 7 Powers Moat Scores")
            m_scores = moat_data["moat_scores"]
            st.bar_chart(pd.DataFrame(list(m_scores.items()), columns=["Power", "Score"]).set_index("Power"))

            st.markdown("### 🔍 4-Quadrant SWOT Matrix")
            swot_col1, swot_col2 = st.columns(2)
            with swot_col1:
                st.success("#### Strengths (Internal)")
                for s in moat_data["swot"]["Strengths"]:
                    st.markdown(f"- {s}")
                st.warning("#### Weaknesses (Internal)")
                for w in moat_data["swot"]["Weaknesses"]:
                    st.markdown(f"- {w}")
            with swot_col2:
                st.info("#### Opportunities (External)")
                for o in moat_data["swot"]["Opportunities"]:
                    st.markdown(f"- {o}")
                st.error("#### Threats (External)")
                for t in moat_data["swot"]["Threats"]:
                    st.markdown(f"- {t}")

            st.markdown("### ⚠️ Feature Parity Trap Detector")
            st.table(pd.DataFrame(moat_data["feature_parity"]))

            st.markdown("### ♟️ Counter-Positioning Moves")
            for move in moat_data["counter_positioning_moves"]:
                st.markdown(f"- {move}")

    # --------------------------------------------------------------------------
    # TAB 6: PM PROMPT SANDBOX & EVALUATOR (PROTOTYPING)
    # --------------------------------------------------------------------------
    with tab_prompts:
        st.markdown("## 🎛️ PM Prompt Sandbox & Evaluation Lab")
        st.markdown("Benchmark system prompts for clarity, hallucination resistance, token cost, and safety guardrails.")

        sample_prompt = (
            "You are a helpful AI assistant for customer support. Summarize user tickets and answer their questions based only on the provided support docs. Never make up refund policies."
        )
        prompt_input = st.text_area("System Prompt to Evaluate:", value=sample_prompt, height=130)

        if st.button("🧪 Evaluate Prompt Rubric", type="primary"):
            eval_res = st.session_state.prompt_eval.evaluate_prompt(prompt_input)
            
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Clarity Score", f"{eval_res['scores']['clarity_score']}/100")
            with e2:
                st.metric("Hallucination Resistance", f"{eval_res['scores']['hallucination_resistance']}/100")
            with e3:
                st.metric("Token Count", eval_res["estimated_tokens"])
            with e4:
                st.metric("Cost / 10k Calls", f"${eval_res['cost_per_10k_calls_usd']}")

            st.markdown("#### 📋 Rubric Checkpoints")
            rubric_cols = st.columns(len(eval_res["rubric_checks"]))
            for idx, (check_name, passed) in enumerate(eval_res["rubric_checks"].items()):
                with rubric_cols[idx]:
                    if passed:
                        st.success(f"✅ {check_name}")
                    else:
                        st.error(f"❌ {check_name}")

            if eval_res["recommendations"]:
                st.markdown("#### 💡 Recommended Improvements")
                for rec in eval_res["recommendations"]:
                    st.markdown(rec)

            st.markdown("#### 🚀 1-Click Optimized Enterprise Prompt Template")
            st.code(eval_res["optimized_prompt_template"], language="markdown")

    # --------------------------------------------------------------------------
    # TAB 7: EXECUTIVE PRD & NARRATIVE CRAFTER (EXPLANATION)
    # --------------------------------------------------------------------------
    with tab_narrative:
        st.markdown("## 📄 Executive Narrative & PRD Crafter")
        st.markdown("The 4th 'E' in the 4E Framework: Explanation. 'Show Before Tell, Touch Before Sell.'")

        doc_type = st.radio("Select Artifact Format:", ["Marty Cagan 1-Pager", "Amazon Working Backwards PR/FAQ"], horizontal=True)
        
        if doc_type == "Marty Cagan 1-Pager":
            c1, c2 = st.columns(2)
            with c1:
                p_name = st.text_input("Product / Initiative:", value="Abhinav's AI PM Studio")
                p_author = st.text_input("Lead PM / Author:", value="Abhinav")
                p_cust = st.text_input("Target Customer:", value="Strategic Product Managers & Founders")
                p_prob = st.text_area("Problem Statement:", value="PMs lack hands-on AI fluency and burn expensive engineering sprints on unvalidated ideas.")
                p_sol = st.text_area("Proposed Solution:", value="A disposable local-first prototyping toolkit that runs 48-hour PoL probes.")
            with c2:
                p_metr = st.text_area("Success Metrics (OKRs):", value="Cut time-to-evidence from 3 weeks to 48 hours; kill >= 30% of bad features before sprint planning.")
                p_val = st.text_input("Value Risk Mitigation:", value="Validated by 80% synthetic persona willingness-to-pay.")
                p_feas = st.text_input("Feasibility Risk Mitigation:", value="Runs on local Ollama or OpenRouter cloud API with zero custom fine-tuning.")
                p_viab = st.text_input("Viability Risk Mitigation:", value="Zero customer data retention ensures complete GDPR/SOC2 compliance.")
                p_evid = st.text_input("Proof-of-Life (PoL) Evidence:", value="Spike-and-delete probe completed in 2 days with 85% stakeholder approval.")

            if st.button("📑 Generate Cagan 1-Pager", type="primary"):
                one_pager_doc = st.session_state.narrative_crafter.generate_cagan_one_pager(
                    p_name, p_cust, p_prob, p_sol, p_metr, p_val, p_feas, p_viab, p_evid, author=p_author
                )
                st.markdown(one_pager_doc)
                st.download_button("📥 Download 1-Pager (Markdown)", data=one_pager_doc, file_name="cagan_1_pager.md", mime="text/markdown")

        else:
            c1, c2 = st.columns(2)
            with c1:
                pr_head = st.text_input("Headline:", value="Abhinav's AI PM Exploration Studio Eliminates Wasted Sprints with Disposable Probes")
                pr_author = st.text_input("Product Lead:", value="Abhinav")
                pr_cname = st.text_input("Customer Name:", value="Sarah Chen")
                pr_ccomp = st.text_input("Customer Company:", value="FinTech Scaleup")
            with c2:
                pr_prob = st.text_input("Problem Addressed:", value="Months wasted debating roadmap priorities without evidence")
                pr_sol = st.text_input("Solution Highlights:", value="Enabling PMs to run 48-hour Proof-of-Life probes with synthetic personas")
                pr_cquote = st.text_area("Customer Quote:", value="Abhinav's toolkit saved us 3 months of engineering work by proving our initial assumption was completely wrong in just 48 hours.")
                pr_equote = st.text_area("Executive Quote:", value="We now require a validated PoL test card before any new AI epic enters sprint planning.")

            if st.button("📰 Generate Amazon PR/FAQ", type="primary"):
                prfaq_doc = st.session_state.narrative_crafter.generate_working_backwards_prfaq(
                    pr_head, pr_cname, pr_ccomp, pr_prob, pr_sol, pr_cquote, pr_equote, author=pr_author
                )
                st.markdown(prfaq_doc)
                st.download_button("📥 Download PR/FAQ (Markdown)", data=prfaq_doc, file_name="amazon_pr_faq.md", mime="text/markdown")

    # --------------------------------------------------------------------------
    # TAB 8: STRATEGIC PM COPILOT
    # --------------------------------------------------------------------------
    with tab_copilot:
        st.markdown("## 💬 Strategic AI PM Co-Pilot")
        st.markdown(f"Collaborate with an AI advisor trained on modern product strategy, paired with **Abhinav** and powered by **OpenRouter**.")

        copilot_mode = st.selectbox(
            "Co-Pilot Persona:",
            ["Marty Cagan (Outcomes & Value Risk)", "Ruthless Prioritizer (Spike-and-Delete)", "Customer Interview Researcher", "Executive Storyteller"]
        )

        # Render conversation history
        for msg in st.session_state.copilot_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        copilot_input = st.chat_input("Ask for advice, prompt ideas, or strategic teardowns...")
        if copilot_input:
            st.session_state.copilot_history.append({"role": "user", "content": copilot_input})
            with st.chat_message("user"):
                st.markdown(copilot_input)

            with st.chat_message("assistant"):
                with st.spinner(f"Consulting {selected_openrouter_model if active_provider == 'openrouter' else active_provider}..."):
                    time.sleep(0.3)
                    sys_prompt = f"You are a world-class Product Management strategic advisor consulting for Abhinav, Lead Product Manager. You are acting in the mode of: {copilot_mode}. Focus on Proof-of-Life probes, the 4E framework (Education, Experimentation, Exploration, Explanation), and evidence over opinion."
                    reply = st.session_state.llm_engine.generate(
                        prompt=copilot_input,
                        system_prompt=sys_prompt,
                        provider=active_provider,
                        model=selected_openrouter_model if active_provider == "openrouter" else None,
                        api_key=api_key_input
                    )
                    st.markdown(reply)
            st.session_state.copilot_history.append({"role": "assistant", "content": reply})


# ==============================================================================
# MODE 2: AI DOCUMENT RAG ASSISTANT (PRESERVED)
# ==============================================================================
else:
    st.markdown(
        """
        <div class="main-header">
            <div class="user-tag">👤 Product Lead: Abhinav</div>
            <h1>📚 Abhinav's AI Document RAG Assistant</h1>
            <p>Upload your documents, generate vector indexes, and query them with transparent citations powered by OpenRouter.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    rag_tab_chat, rag_tab_inspect, rag_tab_guide = st.tabs(["💬 Document Chat", "📑 Vector Inspector", "📖 Guide"])

    with rag_tab_chat:
        with st.expander("📎 Attach & Manage Files", expanded=st.session_state.vector_store.is_empty):
            uploaded_files = st.file_uploader(
                "Upload files (.pdf, .docx, .txt, .csv, .md, .json):",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt", "csv", "md", "json"]
            )
            if uploaded_files:
                for uf in uploaded_files:
                    if uf.name not in [c.source for c in st.session_state.vector_store.chunks]:
                        doc_text = load_document_from_bytes(uf.getvalue(), uf.name)
                        chunks = chunk_documents([doc_text])
                        st.session_state.vector_store.add_chunks(chunks)
                        st.success(f"Indexed {uf.name} ({len(chunks)} chunks)")

        st.markdown(f"**Total Indexed Chunks:** `{len(st.session_state.vector_store.chunks)}`")

        for msg in st.session_state.rag_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        rag_input = st.chat_input("Ask a question about your attached documents...")
        if rag_input:
            st.session_state.rag_messages.append({"role": "user", "content": rag_input})
            with st.chat_message("user"):
                st.markdown(rag_input)

            with st.chat_message("assistant"):
                if st.session_state.vector_store.is_empty:
                    ans = "⚠️ No documents have been attached yet. Please upload files in the Attach & Manage Files section above."
                    st.markdown(ans)
                    st.session_state.rag_messages.append({"role": "assistant", "content": ans})
                else:
                    results = st.session_state.vector_store.similarity_search(rag_input, top_k=3)
                    context_str = "\n\n".join([f"[{c.source}]: {c.content}" for c, _ in results])
                    rag_prompt = f"Context:\n{context_str}\n\nQuestion: {rag_input}\nAnswer citing context:"
                    ans = st.session_state.llm_engine.generate(
                        rag_prompt,
                        provider=active_provider,
                        model=selected_openrouter_model if active_provider == "openrouter" else None,
                        api_key=api_key_input
                    )
                    st.markdown(ans)
                    with st.expander("🔍 View Retrieved Sources"):
                        for c, s in results:
                            st.markdown(f"- **{c.source}** (Score: {s:.2f}): {c.content[:200]}...")
                    st.session_state.rag_messages.append({"role": "assistant", "content": ans})

    with rag_tab_inspect:
        st.markdown("### 📑 Indexed Document Chunks")
        if st.session_state.vector_store.is_empty:
            st.info("No documents indexed yet.")
        else:
            for idx, ch in enumerate(st.session_state.vector_store.chunks[:15], 1):
                with st.expander(f"Chunk #{idx} from {ch.source}"):
                    st.text(ch.content)

    with rag_tab_guide:
        st.markdown(
            """
            ### 📖 Document RAG Quickstart
            1. **Attach Files**: Upload PDF, Word, Markdown, or CSV files.
            2. **Vector Indexing**: Content is split into overlapping chunks and embedded.
            3. **Semantic Querying**: Queries retrieve high-similarity passages to ground responses.
            """
        )

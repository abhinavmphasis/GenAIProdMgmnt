"""
Proof-of-Life (PoL) Probe Framework & Studio
Dean Peters PoL Probes: 5 Flavors of lightweight, disposable reconnaissance missions
to de-risk product decisions before committing engineering resources.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

@dataclass
class PoLProbeTemplate:
    flavor_id: str
    name: str
    tagline: str
    typical_duration: str
    primary_goal: str
    cost_to_run: str
    core_question: str
    common_methods: List[str]
    sample_kill_criteria: str

class PoLProbeStudio:
    """Manages Proof-of-Life probe creation, templates, and evaluation scoring"""

    def __init__(self):
        self.probe_flavors: Dict[str, PoLProbeTemplate] = {
            "feasibility_check": PoLProbeTemplate(
                flavor_id="feasibility_check",
                name="1. Feasibility Check (Spike-and-Delete)",
                tagline="Test if the AI or tech can actually do the job in 24–48 hours",
                typical_duration="1 - 2 Days",
                primary_goal="Expose lethal technical hurdles and model capability ceilings",
                cost_to_run="Near-zero engineering sprint impact",
                core_question="Can the model reliably solve the problem without brittle custom hacks?",
                common_methods=[
                    "Scripted throwaway script with off-the-shelf model",
                    "Prompt playground edge-case stress test",
                    "Latency and cost per invocation benchmark",
                    "Synthetic edge-case evaluation"
                ],
                sample_kill_criteria="If prompt latency > 3.5s or accuracy < 85% on basic sample inputs, kill before sprint planning."
            ),
            "demand_smoke_test": PoLProbeTemplate(
                flavor_id="demand_smoke_test",
                name="2. Demand Smoke Test (Painted Door)",
                tagline="Test genuine user appetite before writing a single line of backend code",
                typical_duration="2 - 4 Days",
                primary_goal="Verify user intent, click-through urgency, and willingness-to-act",
                cost_to_run="Low (static UI mockup or dummy toggle)",
                core_question="Do real users actively try to use this capability when prompted?",
                common_methods=[
                    "In-app 'Coming Soon' toggle with email notification capture",
                    "Lightweight Typeform or single-page landing probe",
                    "Email newsletter feature interest link",
                    "Customer interview manual concierge trial"
                ],
                sample_kill_criteria="If CTR on the prototype button is under 8% among active power users, kill feature."
            ),
            "usability_smoke_test": PoLProbeTemplate(
                flavor_id="usability_smoke_test",
                name="3. Usability Smoke Test (Micro-Prototype)",
                tagline="Evaluate cognitive friction and trust in AI outputs",
                typical_duration="2 - 3 Days",
                primary_goal="Discover if users understand, trust, and can act on the AI results",
                cost_to_run="Low (Clickable mockup or Streamlit probe)",
                core_question="Do users trust the output and intuitively know how to correct errors?",
                common_methods=[
                    "Streamlit interactive sandbox with sample datasets",
                    "Paper or Penpot wireframe walk-through with 5 target users",
                    "Wizard of Oz simulation (PM acts as model backend)",
                    "Side-by-side output critique interview"
                ],
                sample_kill_criteria="If >= 3 out of 5 users cannot explain why the AI gave its recommendation, redesign UX."
            ),
            "data_feasibility_audit": PoLProbeTemplate(
                flavor_id="data_feasibility_audit",
                name="4. Data Feasibility Audit",
                tagline="Verify if high-quality signal data actually exists",
                typical_duration="1 - 3 Days",
                primary_goal="Audit data cleanliness, completeness, privacy constraints, and vectorizability",
                cost_to_run="Zero dev build cost (pure exploratory query)",
                core_question="Do we have enough clean, labeled, compliant data to ground the AI?",
                common_methods=[
                    "Direct database sample export and null-value distribution check",
                    "Chunking & embedding similarity distribution test",
                    "PII / GDPR / compliance privacy scan",
                    "Synthetic data augmentation baseline comparison"
                ],
                sample_kill_criteria="If > 40% of target documents lack key entity metadata, pause AI feature."
            ),
            "narrative_pitch_probe": PoLProbeTemplate(
                flavor_id="narrative_pitch_probe",
                name="5. Executive Narrative Pitch Probe",
                tagline="Touch Before Sell: Win leadership alignment with interactive evidence",
                typical_duration="1 Day",
                primary_goal="De-risk stakeholder skepticism and establish shared definition of success",
                cost_to_run="Extremely low (Interactive 1-Pager + live clickable probe)",
                core_question="Will executive leadership back this initiative with budget and headcount?",
                common_methods=[
                    "Interactive Streamlit prototype demo during executive check-in",
                    "Working Backwards Press Release (Amazon PR/FAQ)",
                    "Show-Before-Tell risk-mitigation scorecard",
                    "Recorded Loom walk-through with synthetic persona feedback"
                ],
                sample_kill_criteria="If executive sponsors do not agree on the kill metric within 20 minutes, iterate narrative."
            )
        }
        
        self.preset_scenarios = {
            "ai_copilot": {
                "name": "AI Workflow Copilot for Support Agents",
                "hypothesis": "Providing real-time response suggestions based on past solved tickets will reduce agent handle time by 30%.",
                "risk_area": "Model hallucinating inaccurate policies or generating robotic answers",
                "cheapest_probe": "feasibility_check",
                "kill_criteria": "If accuracy is below 90% on top 20 historical ticket types, do not integrate into CRM."
            },
            "smart_search": {
                "name": "Natural Language Document RAG Search",
                "hypothesis": "Enabling natural language Q&A across company documentation will save employees 45 mins/week.",
                "risk_area": "Data freshness and embedding chunk quality",
                "cheapest_probe": "data_feasibility_audit",
                "kill_criteria": "If > 30% of search queries retrieve outdated or deprecated internal docs, kill until docs are audited."
            },
            "automated_summary": {
                "name": "Automated Customer Interview Synthesis",
                "hypothesis": "PMs will adopt an automated audio extraction tool to draft user stories in < 5 minutes.",
                "risk_area": "PMs not trusting automated extraction and reviewing entire audio anyway",
                "cheapest_probe": "usability_smoke_test",
                "kill_criteria": "If PMs spend more than 10 minutes verifying each transcript, the probe fails."
            }
        }

    def calculate_priority_score(
        self,
        impact_score: int,       # 1-10 (How much value if it works)
        risk_uncertainty: int,   # 1-10 (How high is the risk / unknown)
        effort_probe: int,       # 1-10 (How fast/cheap is the probe to run; lower is cheaper)
        strategic_alignment: int # 1-10 (Alignment with company OKRs)
    ) -> Dict[str, Any]:
        """
        Calculates the Proof-of-Life Probe Priority Score.
        High Risk + High Impact + Cheap to Probe = Highest Priority to run FIRST.
        """
        # Weighted PoL Index:
        # High uncertainty warrants high probe priority so we kill or confirm early
        urgency_index = ((risk_uncertainty * 1.5) + (impact_score * 1.2) + (strategic_alignment * 1.0)) / (effort_probe * 0.8 + 1)
        normalized_score = min(100.0, max(5.0, urgency_index * 12.5))
        
        if normalized_score >= 75:
            recommendation = "🚨 URGENT PROBE: Run this spike-and-delete test immediately before any roadmap lock-in."
            badge = "HIGHEST PRIORITY"
        elif normalized_score >= 50:
            recommendation = "⚡ SCHEDULE PROBE: High value, moderate uncertainty. Allocate 2 days for lightweight test."
            badge = "MEDIUM-HIGH PRIORITY"
        else:
            recommendation = "⏳ DE-PRIORITIZE: Low uncertainty or high probe effort. Address higher-risk unknowns first."
            badge = "LOW PRIORITY"
            
        return {
            "score": round(normalized_score, 1),
            "badge": badge,
            "recommendation": recommendation,
            "metrics": {
                "impact": impact_score,
                "uncertainty": risk_uncertainty,
                "probe_effort": effort_probe,
                "alignment": strategic_alignment
            }
        }

    def generate_test_card(
        self,
        feature_name: str,
        hypothesis: str,
        probe_flavor: str,
        lethal_assumption: str,
        success_criteria: str,
        kill_criteria: str,
        duration_days: int = 2,
        lead_pm: str = "Abhinav"
    ) -> str:
        """Format an exportable Proof-of-Life Test Card"""
        flavor_info = self.probe_flavors.get(probe_flavor, list(self.probe_flavors.values())[0])
        
        test_card = f"""# 🧪 PROOF-OF-LIFE (PoL) PROBE TEST CARD
**Feature / Concept:** {feature_name}  
**Lead Product Manager:** {lead_pm}  
**Probe Flavor:** {flavor_info.name}  
**Execution Window:** {duration_days} Days (Spike-and-Delete)  
**Cost Threshold:** Near-zero dev bandwidth (PM-driven exploration)  

---

### 1. CORE HYPOTHESIS
> **We believe that:**  
> {hypothesis}

### 2. THE LETHAL ASSUMPTION (What kills this idea if false?)
> **Lethal Assumption:**  
> {lethal_assumption}

### 3. THE CHEAPEST PROBE
- **Probe Category:** {flavor_info.name}
- **Primary Goal:** {flavor_info.primary_goal}
- **Methodology to Use:**
{chr(10).join([f"  - [ ] {m}" for m in flavor_info.common_methods])}

### 4. EVIDENCE THRESHOLDS (Passing vs. Kill Criteria)
- **✅ PASS CRITERIA (Confidence to proceed):**  
  {success_criteria}
- **🛑 KILL CRITERIA (Pull the plug immediately):**  
  {kill_criteria}

---
*Philosophical Reminder:* "If the prototype doesn't have the potential to sting, it's just theater. Celebrate killing bad ideas before sprint planning!"
"""
        return test_card

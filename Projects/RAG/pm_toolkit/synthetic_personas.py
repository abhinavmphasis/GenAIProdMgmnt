"""
Synthetic Persona & Market "Wind Tunnel" Simulator
Generates rich synthetic customer profiles and simulates their reactions
to proposed feature concepts before spending time or money talking to real users.
"""

import json
import random
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class Persona:
    id: str
    name: str
    title: str
    company_type: str
    industry: str
    company_size: str
    tech_savviness: int  # 1-10
    budget_authority_usd: int
    primary_kpi: str
    pain_points: List[str]
    goals: List[str]
    objection_tendency: str

class SyntheticPersonaGenerator:
    """Generates synthetic personas and runs wind-tunnel feature reactions"""

    def __init__(self):
        self.industry_templates = {
            "b2b_saas": {
                "names": ["Elena Vance", "Marcus Reed", "Sarah Chen", "Devon Miller", "Priya Nair", "Liam O'Connor"],
                "titles": ["Head of Product", "VP of Engineering", "Customer Success Lead", "Director of RevOps", "Chief Growth Officer"],
                "company_types": ["Series B B2B SaaS", "Enterprise Software", "Mid-Market CRM", "High-Growth Scaleup"],
                "company_sizes": ["50-200 employees", "200-1000 employees", "1000+ employees"],
                "kpis": ["Net Dollar Retention (NDR)", "Feature Adoption %", "Gross Margin", "CAC Payback Period"],
                "pain_points": [
                    "Too many fragmented SaaS tools that don't speak to each other",
                    "Leadership wants AI features launched yesterday without clear business value",
                    "Engineers spend 40% of their week on support bug tickets instead of core product",
                    "Customer feedback is scattered across Slack, Gong, Zendesk, and Notion",
                    "Struggling to calculate ROI on new third-party software subscriptions"
                ],
                "goals": [
                    "Consolidate team workflows to accelerate sprint velocity",
                    "Automate manual weekly cross-departmental reporting",
                    "De-risk big product bets with rapid prototyping",
                    "Improve onboarding completion rate by 20%"
                ],
                "objections": ["Worried about security & data leakage", "Too complex to integrate with our Jira/Salesforce", "Team doesn't have time to learn another tool"]
            },
            "b2c_consumer": {
                "names": ["Jordan Taylor", "Chloe Bennett", "Alex Rivera", "Samira Khan", "Mateo Rossi", "Zoe Mitchell"],
                "titles": ["Freelance Creator", "Digital Nomad / Consultant", "Busy Parent & Marketer", "Graduate Student", "Fitness Community Lead"],
                "company_types": ["Solo Operator", "D2C Brand", "Consumer Marketplace", "Lifestyle Creator"],
                "company_sizes": ["Self-employed", "1-10 employees", "Small Community"],
                "kpis": ["Daily Active Engagement", "Monthly Subscription Budget", "Time Saved Per Day", "Stress Reduction"],
                "pain_points": [
                    "Subscription fatigue — tired of paying $15/month for single-feature apps",
                    "Invasive notifications and aggressive upsell popups",
                    "Privacy concerns over personal photos and voice data",
                    "Mobile apps that require too many onboarding steps before value"
                ],
                "goals": [
                    "Get instant value in under 60 seconds without tutorials",
                    "Keep personal information private and offline",
                    "Simplify daily task tracking without feeling overwhelmed"
                ],
                "objections": ["Will this cost me another recurring subscription?", "Why does this need my contact list?", "Seems like an over-engineered solution for a simple problem"]
            },
            "fintech": {
                "names": ["Arthur Pendelton", "Siddharth Verma", "Nadia Al-Mansoor", "Grace Hopper-Lee", "David Morales"],
                "titles": ["Risk & Compliance Officer", "Head of Treasury Operations", "WealthTech PM", "Fintech Founder"],
                "company_types": ["Digital Neobank", "Payment Gateway", "Commercial Lending Platform", "Crypto Custody Provider"],
                "company_sizes": ["100-500 employees", "500-2500 employees"],
                "kpis": ["Fraud Loss Rate", "Transaction Processing Latency", "Regulatory Audit Pass Rate", "AUM Growth"],
                "pain_points": [
                    "Strict audit requirements make adopting new AI models a compliance nightmare",
                    "Legacy core banking infrastructure with SOAP/XML endpoints",
                    "Model hallucination in financial advice creates lethal legal liabilities",
                    "Reconciliation discrepancies between internal ledgers and payment rails"
                ],
                "goals": [
                    "Automate compliance reporting without risking regulatory fines",
                    "Speed up KYC/AML verification from hours to seconds",
                    "Protect customer PII with zero data retention policies"
                ],
                "objections": ["How is this SOC2 Type II and GDPR compliant?", "Can we run this fully on-premises?", "What is the liability if the model gives wrong advice?"]
            },
            "healthtech": {
                "names": ["Dr. Ananya Sharma", "Julian Beck", "Rachel Greenwald", "Kenneth Washington"],
                "titles": ["Chief Medical Information Officer", "Clinical Operations Lead", "Telehealth Product Manager", "Patient Experience Director"],
                "company_types": ["Hospital System", "Virtual Care Provider", "Electronic Health Record (EHR) Platform", "Diagnostic AI Startup"],
                "company_sizes": ["250-5000 employees"],
                "kpis": ["Clinician Burnout Index", "Patient Wait Times", "HIPAA Compliance 100%", "Diagnostic Concordance"],
                "pain_points": [
                    "Doctors spend 2+ hours charting after hours ('pajama time')",
                    "EHR integration is painfully archaic and expensive (HL7/FHIR hurdles)",
                    "High clinical skepticism of automated recommendations",
                    "Strict HIPAA privacy laws on patient audio & health records"
                ],
                "goals": [
                    "Eliminate manual doctor documentation using ambient voice AI",
                    "Safely summarize medical histories with verified clinical citations",
                    "Reduce patient no-show rates through intelligent reminders"
                ],
                "objections": ["Is this HIPAA compliant with signed BAA?", "What happens when an edge case is missed?", "Will physicians reject this as another screen to click?"]
            },
            "ai_agents": {
                "names": ["Tate Lindqvist", "Hana Takahashi", "Sean Brody", "Camila Silva"],
                "titles": ["AI Solutions Architect", "Founding AI Engineer", "Agent Systems PM", "Developer Advocate"],
                "company_types": ["Autonomous Agent Platform", "Developer Tooling Startup", "Enterprise GenAI Lab", "Open-Source AI Collective"],
                "company_sizes": ["10-100 employees"],
                "kpis": ["Agent Task Completion Rate", "Cost Per Task Resolved", "API Error Rate", "Latency to First Action"],
                "pain_points": [
                    "Agents getting stuck in infinite tool-calling loops",
                    "High token bills with low completion fidelity",
                    "Lack of deterministic evals for non-deterministic agents",
                    "Opaque prompt drift across model updates"
                ],
                "goals": [
                    "Build reliable multi-step agent workflows with strict guardrails",
                    "Cut inference costs by 50% using model routing and caching",
                    "Implement regression testing before deploying agent updates"
                ],
                "objections": ["Does this support local models via Ollama?", "Can I export the trace telemetry to Langfuse?", "Can we bring our own API keys?"]
            }
        }

    def generate_personas(self, industry: str = "b2b_saas", count: int = 5) -> List[Persona]:
        """Generate a specified number of diverse synthetic personas"""
        tmpl = self.industry_templates.get(industry, self.industry_templates["b2b_saas"])
        personas = []
        
        for i in range(count):
            name = tmpl["names"][i % len(tmpl["names"])]
            title = tmpl["titles"][i % len(tmpl["titles"])]
            company = tmpl["company_types"][i % len(tmpl["company_types"])]
            size = tmpl["company_sizes"][i % len(tmpl["company_sizes"])]
            kpi = tmpl["kpis"][i % len(tmpl["kpis"])]
            
            p_p = random.sample(tmpl["pain_points"], min(2, len(tmpl["pain_points"])))
            gls = random.sample(tmpl["goals"], min(2, len(tmpl["goals"])))
            obj = tmpl["objections"][i % len(tmpl["objections"])]
            
            budget = random.choice([5000, 15000, 50000, 120000, 250000]) if "b2b" in industry or "fintech" in industry else random.choice([10, 25, 50, 100])
            tech = random.randint(6, 10) if industry == "ai_agents" else random.randint(3, 9)
            
            p = Persona(
                id=f"persona_{industry[:3]}_{i+1:02d}",
                name=f"{name}",
                title=title,
                company_type=company,
                industry=industry.replace("_", " ").title(),
                company_size=size,
                tech_savviness=tech,
                budget_authority_usd=budget,
                primary_kpi=kpi,
                pain_points=p_p,
                goals=gls,
                objection_tendency=obj
            )
            personas.append(p)
            
        return personas

    def run_wind_tunnel_simulation(self, feature_proposal: str, personas: List[Persona]) -> Dict[str, Any]:
        """
        Simulates how generated personas react to a feature proposal.
        Returns sentiment breakdown, acceptance rate, objections, and willingness-to-pay.
        """
        results = []
        enthusiastic = 0
        skeptical = 0
        opposed = 0
        
        feature_lower = feature_proposal.lower()
        
        for p in personas:
            # Deterministic/Heuristic evaluation based on persona characteristics
            tech = p.tech_savviness
            objection = p.objection_tendency
            
            # Check alignment with pain points
            matches_pain = any(any(word in p_pain.lower() for word in feature_lower.split() if len(word) > 4) for p_pain in p.pain_points) if p.pain_points else False
            
            if matches_pain or tech >= 8:
                sentiment = "Enthusiastic"
                enthusiastic += 1
                wtp_multiplier = 1.2
                reaction_quote = f"If this reliably solves our issue with {p.pain_points[0][:40]}..., I would pilot this tomorrow."
            elif tech <= 4 or "security" in objection.lower() or "compliance" in objection.lower():
                sentiment = "Opposed"
                opposed += 1
                wtp_multiplier = 0.0
                reaction_quote = f"Hard pass right now. My biggest red flag: {objection}"
            else:
                sentiment = "Skeptical"
                skeptical += 1
                wtp_multiplier = 0.7
                reaction_quote = f"Interesting concept, but {objection}. Show me concrete evidence first."
                
            est_wtp = round(p.budget_authority_usd * 0.05 * wtp_multiplier) if p.budget_authority_usd > 500 else round(20 * wtp_multiplier)
            
            results.append({
                "persona_id": p.id,
                "name": p.name,
                "title": p.title,
                "company": p.company_type,
                "sentiment": sentiment,
                "simulated_quote": reaction_quote,
                "primary_objection": objection,
                "estimated_wtp_usd": est_wtp
            })
            
        acceptance_rate = round((enthusiastic / len(personas)) * 100, 1) if personas else 0
        
        return {
            "feature_analyzed": feature_proposal,
            "total_simulated": len(personas),
            "acceptance_rate_pct": acceptance_rate,
            "sentiment_summary": {
                "Enthusiastic": enthusiastic,
                "Skeptical": skeptical,
                "Opposed": opposed
            },
            "key_objection_patterns": list(set(r["primary_objection"] for r in results)),
            "detailed_reactions": results
        }

    def personas_to_dataframe(self, personas: List[Persona]) -> pd.DataFrame:
        """Convert persona list into pandas DataFrame for Streamlit display & CSV export"""
        records = []
        for p in personas:
            records.append({
                "ID": p.id,
                "Name": p.name,
                "Title": p.title,
                "Company": p.company_type,
                "Size": p.company_size,
                "Savviness (1-10)": p.tech_savviness,
                "Budget ($)": f"${p.budget_authority_usd:,}",
                "Primary KPI": p.primary_kpi,
                "Top Pain Point": p.pain_points[0] if p.pain_points else "None",
                "Primary Objection": p.objection_tendency
            })
        return pd.DataFrame(records)

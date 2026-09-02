"""
Competitive Intelligence & Moat Analyzer
Helps PMs assess competitor vulnerabilities, evaluate 7 Powers Moat strength,
and design counter-positioning moves before committing engineering time.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class CompetitorProfile:
    name: str
    category: str
    target_segment: str
    core_strength: str
    glaring_weakness: str
    pricing_model: str

class CompetitiveIntelligenceLab:
    """Competitive teardowns, SWOT generator, and 7 Powers moat analyzer"""

    def __init__(self):
        self.preset_competitors = {
            "Enterprise Legacy Incumbent (e.g. Jira / Salesforce)": CompetitorProfile(
                name="Legacy Enterprise Incumbent",
                category="Enterprise B2B Software",
                target_segment="Fortune 500 Procurement & IT Leaders",
                core_strength="Massive switching costs, existing vendor contracts, and deep integrations",
                glaring_weakness="Sluggish, bloated UI, complex configuration overhead, zero offline support",
                pricing_model="Expensive per-seat annual enterprise contract with mandatory add-ons"
            ),
            "AI-Native Point Solution (e.g. Linear / Cursor / Perplexity)": CompetitorProfile(
                name="AI-Native Point Solution",
                category="Next-Gen Developer/PM Tools",
                target_segment="Product-Led Growth, High-Velocity Engineers & Startups",
                core_strength="Blazing fast UI speed, keyboard shortcuts, opinionated modern workflows",
                glaring_weakness="Limited enterprise governance, lacks complex cross-department reporting",
                pricing_model="Bottom-up freemium with $15-30/user/month self-serve tier"
            ),
            "All-in-One Collaboration Suite (e.g. Notion / ClickUp)": CompetitorProfile(
                name="All-in-One Collaboration Suite",
                category="Workspace & Knowledge OS",
                target_segment="Cross-functional SMB & Mid-Market Teams",
                core_strength="Infinite flexibility, docs + databases + AI in one canvas",
                glaring_weakness="Jack of all trades, master of none; degrades into messy wikis without strict hygiene",
                pricing_model="Freemium with modular AI add-on tier ($10/user/month)"
            )
        }

    def analyze_moat_and_vulnerabilities(
        self,
        company_name: str,
        competitor_name: str,
        market_sector: str = "B2B SaaS"
    ) -> Dict[str, Any]:
        """
        Calculates Hamilton Helmer 7 Powers Moat scores and strategic counter-positioning.
        """
        # 7 Powers Moat Scoring (1-10)
        moat_scores = {
            "Switching Costs": 8,
            "Network Effects": 5,
            "Counter-Positioning": 9,
            "Scale Economies": 6,
            "Cornered Resource": 4,
            "Brand Affinity": 7,
            "Process Power": 6
        }
        
        swot = {
            "Strengths": [
                f"High-velocity development cycle compared to {competitor_name}",
                "Local-first privacy and zero customer data lock-in",
                "Lightweight, disposable experimentation workflow"
            ],
            "Weaknesses": [
                f"Smaller brand recognition compared to {competitor_name}",
                "Fewer out-of-the-box legacy enterprise connectors",
                "Self-serve onboarding requires higher initial user initiative"
            ],
            "Opportunities": [
                f"Capitalize on customer backlash against {competitor_name}'s recent pricing hikes",
                "Win developers and technical PMs with local LLM (Ollama) privacy capabilities",
                "Provide transparent AI citations while competitors provide black-box answers"
            ],
            "Threats": [
                f"{competitor_name} bundling similar AI features for free into existing enterprise plans",
                "Model provider APIs changing pricing or rate limits suddenly",
                "Enterprise procurement demanding 12-month SOC2 certification upfront"
            ]
        }
        
        feature_parity = [
            {"feature": "Core Workflow Execution", "our_status": "✅ Streamlined & Fast", "competitor_status": "⚠️ Clunky / Complex", "recommendation": "Lean into speed"},
            {"feature": "Local Offline Support", "our_status": "✅ 100% Offline Capable", "competitor_status": "❌ Cloud Only", "recommendation": "Differentiating Moat"},
            {"feature": "Legacy SAML / SSO", "our_status": "⏳ Roadmap / Phase 2", "competitor_status": "✅ Enterprise Native", "recommendation": "Address when closing >$50k deals"},
            {"feature": "Custom Reporting Dashboard", "our_status": "⚠️ Basic Markdown Export", "competitor_status": "✅ 50+ Custom Charts", "recommendation": "Trap Feature: Do not clone 50 charts!"}
        ]
        
        counter_moves = [
            f"**Play the 'David vs Goliath' Speed Card**: Position our tool as a 2-day proof-of-life probe engine, while {competitor_name} takes 3 months to configure.",
            "**Leverage Privacy as a Wedge**: Emphasize zero-retention and local Ollama capability for compliance-sensitive prospects.",
            "**Avoid the 'Feature Parity Trap'**: Do not build the bottom 80% of legacy bells and whistles that customers rarely touch."
        ]
        
        return {
            "target_company": company_name,
            "competitor_analyzed": competitor_name,
            "market_sector": market_sector,
            "moat_scores": moat_scores,
            "swot": swot,
            "feature_parity": feature_parity,
            "counter_positioning_moves": counter_moves
        }

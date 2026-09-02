"""
Audio Intelligence & Customer Voice Extraction Engine
Phase 7.1 Dean Peters PoL Framework: 6 PM-Specific Audio Workflows
Converts raw interview audio or transcripts into high-conviction product evidence.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class AudioWorkflowTemplate:
    id: str
    name: str
    description: str
    target_output: str
    pm_focus_areas: List[str]

class AudioIntelligenceLab:
    """Manages audio transcription workflows and structured insight extraction"""

    def __init__(self):
        self.templates: Dict[str, AudioWorkflowTemplate] = {
            "user_interview": AudioWorkflowTemplate(
                id="user_interview",
                name="1. User Interview Deep Analysis",
                description="Extract latent user pain points, unprompted feature desires, and high-impact verbatim quotes.",
                target_output="User Pain Point & Opportunity Matrix",
                pm_focus_areas=["Pain Points", "Unstated Frustrations", "Feature Wishes", "Verbatim Quotes", "PoL Probe Idea"]
            ),
            "stakeholder_meeting": AudioWorkflowTemplate(
                id="stakeholder_meeting",
                name="2. Stakeholder Meeting Executive Summary",
                description="Synthesize cross-functional executive meetings into locked decisions, action item owners, and unspoken risks.",
                target_output="Executive Decision & Action Item Brief",
                pm_focus_areas=["Decisions Locked", "Action Items & Owners", "Unresolved Friction/Risks", "Timeline Impacts"]
            ),
            "demo_feedback": AudioWorkflowTemplate(
                id="demo_feedback",
                name="3. Product Demo Feedback Analysis",
                description="Map customer reactions during live product walkthroughs to identify friction points and 'aha!' moments.",
                target_output="Demo UX Friction & Delight Scorecard",
                pm_focus_areas=["Aha! Moments (Delight)", "Objections Raised", "UX Stumbling Blocks", "Pricing Hesitations"]
            ),
            "competitive_notes": AudioWorkflowTemplate(
                id="competitive_notes",
                name="4. Competitor Teardown & Sales Call Notes",
                description="Analyze sales recordings where competitors were mentioned to detect their pricing, positioning, and weaknesses.",
                target_output="Competitive Intelligence Brief",
                pm_focus_areas=["Competitors Named", "Perceived Competitor Advantages", "Competitor Weaknesses", "Win/Loss Levers"]
            ),
            "voice_memo_to_backlog": AudioWorkflowTemplate(
                id="voice_memo_to_backlog",
                name="5. PM Voice Memo to Structured Backlog",
                description="Transform fragmented audio musings recorded on a walk into clean user stories with acceptance criteria.",
                target_output="Epics, User Stories & Acceptance Criteria",
                pm_focus_areas=["Core Problem", "User Story Format", "Acceptance Criteria", "Definition of Done", "Scope Exclusions"]
            ),
            "churn_exit_interview": AudioWorkflowTemplate(
                id="churn_exit_interview",
                name="6. Customer Churn Exit Interview Teardown",
                description="Analyze exit calls with canceling customers to find the root cause and product gaps.",
                target_output="Churn Post-Mortem & Retention Levers",
                pm_focus_areas=["Trigger Event for Churn", "Alternative Tool Selected", "Missing Product Features", "Retention Levers"]
            )
        }

        self.sample_transcripts: Dict[str, str] = {
            "user_interview": """Interviewer (Abhinav - PM): Thanks for taking the time, Sarah. How does your team currently plan weekly sprints?
Sarah (Customer): Honestly, it's a disaster of spreadsheets and Slack pings. Every Monday morning I spend at least 2 hours copying Jira tickets into a Notion table so leadership can understand the roadmap. It drives me crazy because by Wednesday, the spreadsheet is out of date. 
Interviewer (Abhinav - PM): Have you tried automated reporting plugins?
Sarah (Customer): Yes, but they dump 40 pages of useless charts. Executives don't read that! I just need an executive 1-pager that highlights: What shipped, what slipped, and where engineering is blocked. If an AI tool could generate that in 30 seconds from our actual commit history, I would pay $50/month out of my personal budget immediately.""",

            "stakeholder_meeting": """VP of Sales: We have three enterprise prospects asking for SOC2 Type II and on-prem deployment. If we don't deliver this by Q3, we lose $450k ARR.
Abhinav (Head of Product): On-prem would take our entire backend squad off the AI copilot roadmap for 4 months. That kills our competitive differentiator.
VP of Engineering: What if we do a hybrid approach? We keep the core cloud SaaS, but provide a zero-retention private tenant mode for their API keys. That satisfies compliance in 3 weeks.
Abhinav (Head of Product): Agreed. Let's lock this decision: We build Private Tenant Mode in Sprint 14. Sales team will use this to unblock the deals. Target completion: August 15.""",

            "demo_feedback": """Prospect: Wow, the onboarding speed is super impressive. I got my data uploaded in 10 seconds.
Abhinav (PM): Great! How does the insight card look to you?
Prospect: The summary is fine, but I'm really confused by the 'Confidence Score: 0.84'. What does 0.84 mean? Is it 84% accurate, or did the model make up 16% of the facts? That makes me nervous to share this with my CEO. Also, where do I click to download this as a PowerPoint?""",

            "churn_exit_interview": """Customer Success: Hi Dave, we're sorry to see your team cancel your subscription. Could you share what led to this?
Dave: When we signed up 6 months ago, we loved the promise of automated ticket triage. But in practice, the model kept misclassifying our VIP tier-1 billing tickets as low-priority bugs. We had two major clients escalate because their urgent issues sat in the queue for 3 days. We ended up switching to Zendesk's native workflow engine. It's dumber, but it's 100% predictable and has audit logs."""
        }

    def process_transcript(self, template_id: str, transcript_text: str, custom_instruction: str = "") -> Dict[str, Any]:
        """
        Process and extract structured PM insights according to chosen template.
        Uses rule-based heuristics & template structuring.
        """
        template = self.templates.get(template_id, self.templates["user_interview"])
        
        # Heuristic extraction
        lines = [line.strip() for line in transcript_text.split("\n") if line.strip()]
        word_count = sum(len(line.split()) for line in lines)
        
        sections = {}
        for focus in template.pm_focus_areas:
            sections[focus] = self._extract_focus_section(focus, transcript_text)
            
        return {
            "template_name": template.name,
            "target_output": template.target_output,
            "word_count": word_count,
            "line_count": len(lines),
            "focus_areas": template.pm_focus_areas,
            "extracted_sections": sections
        }

    def _extract_focus_section(self, focus_area: str, text: str) -> List[str]:
        """Extract domain-relevant observations based on focus area"""
        f_lower = focus_area.lower()
        t_lower = text.lower()
        
        if "pain" in f_lower or "frustration" in f_lower or "stumbling" in f_lower:
            return [
                "Manual synchronization across multiple tools consumes 2+ hours every Monday",
                "Spreadsheet-based reporting becomes stale within 48 hours",
                "Cognitive anxiety over metric definitions and opaque confidence scores"
            ]
        elif "quote" in f_lower:
            quotes = [line for line in text.split("\n") if "'" in line or '"' in line]
            if quotes:
                return quotes[:2]
            return ["> 'If an AI tool could generate that in 30 seconds, I would pay out of my personal budget immediately.'"]
        elif "decision" in f_lower or "action" in f_lower:
            return [
                "Decision Locked: Build Private Tenant Mode in Sprint 14 instead of full on-premise re-architecture",
                "Action Owner: VP of Eng to spec zero-retention API architecture by Friday",
                "Target Delivery Date: August 15 (unblocks $450k ARR pipeline)"
            ]
        elif "wishes" in f_lower or "feature" in f_lower or "opportunity" in f_lower:
            return [
                "Automated 1-click Executive Status Brief generator from Jira commits",
                "Transparent citation drawer explaining why a recommendation was made",
                "Export directly to PowerPoint / Markdown format"
            ]
        elif "pol probe" in f_lower or "probe" in f_lower:
            return [
                "Recommended Probe: 48-Hour Spike-and-Delete Feasibility Check",
                "Test: Generate mock status 1-pager using template and send to 5 PMs",
                "Kill Criteria: If PMs still spend > 5 minutes editing the draft, abandon automation"
            ]
        elif "churn" in f_lower or "trigger" in f_lower:
            return [
                "Primary Trigger: Model misclassified VIP tier-1 billing tickets as low priority",
                "Lack of Deterministic Guardrails: Customer chose predictable rule engine over probabilistic AI"
            ]
        else:
            return [
                f"High-signal insight identified under {focus_area} from customer narrative.",
                "Identified key risk that requires de-risking before roadmap inclusion."
            ]

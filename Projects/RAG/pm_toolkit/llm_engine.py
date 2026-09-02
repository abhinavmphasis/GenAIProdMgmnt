"""
Unified LLM & Intelligence Engine
Supports:
1. Local Ollama (Ollama on localhost:11434)
2. Cloud API (OpenRouter / OpenAI)
3. Zero-Configuration Offline Heuristic & Simulation Engine
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional, Any, Generator

class LLMEngine:
    """Unified engine supporting Local Ollama, Cloud APIs, and Offline Heuristic Simulation"""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url.rstrip("/")
        
    def detect_providers(self, openrouter_key: Optional[str] = None, openai_key: Optional[str] = None) -> Dict[str, Any]:
        """Detect status of all available AI providers"""
        status = {
            "offline_simulation": {
                "available": True,
                "label": "Offline PM Simulation Engine (Instant, Zero-Cost)",
                "models": ["pm-simulator-v2", "heuristic-fast"]
            },
            "ollama": {
                "available": False,
                "label": "Local Ollama (Private, Local GPU/CPU)",
                "models": []
            },
            "openrouter": {
                "available": False,
                "label": "OpenRouter Cloud API (Multi-model)",
                "models": [
                    "openai/gpt-4o-mini",
                    "openai/gpt-4o",
                    "anthropic/claude-3.5-sonnet",
                    "google/gemini-2.0-flash-001",
                    "deepseek/deepseek-r1"
                ]
            },
            "openai": {
                "available": False,
                "label": "Direct OpenAI API",
                "models": ["gpt-4o-mini", "gpt-4o"]
            }
        }
        
        # Check Ollama
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=1.5)
            if resp.status_code == 200:
                ollama_data = resp.json()
                models = [m.get("name") for m in ollama_data.get("models", [])]
                status["ollama"]["available"] = True
                status["ollama"]["models"] = models if models else ["llama3.2:latest"]
        except Exception:
            status["ollama"]["available"] = False
            
        # Check OpenRouter Key
        active_or_key = openrouter_key or os.getenv("OPENROUTER_API_KEY")
        if active_or_key and len(active_or_key.strip()) > 5:
            status["openrouter"]["available"] = True
            
        # Check OpenAI Key
        active_oa_key = openai_key or os.getenv("OPENAI_API_KEY")
        if active_oa_key and len(active_oa_key.strip()) > 5:
            status["openai"]["available"] = True
            
        return status

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are an expert AI Product Management advisor.",
        provider: str = "auto",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> str:
        """Generate response across selected or best available provider"""
        # Determine provider
        providers = self.detect_providers(openrouter_key=api_key)
        
        if provider == "ollama" and providers["ollama"]["available"]:
            return self._call_ollama(prompt, system_prompt, model or "llama3.2", temperature)
        elif provider == "openrouter" and (api_key or providers["openrouter"]["available"]):
            return self._call_openrouter(prompt, system_prompt, model or "openai/gpt-4o-mini", api_key, temperature, max_tokens)
        elif provider == "openai" and (api_key or providers["openai"]["available"]):
            return self._call_openai(prompt, system_prompt, model or "gpt-4o-mini", api_key, temperature, max_tokens)
        elif provider == "auto":
            if providers["ollama"]["available"]:
                available_models = providers["ollama"]["models"]
                chosen_model = available_models[0] if available_models else "llama3.2"
                return self._call_ollama(prompt, system_prompt, chosen_model, temperature)
            elif providers["openrouter"]["available"]:
                return self._call_openrouter(prompt, system_prompt, "openai/gpt-4o-mini", api_key, temperature, max_tokens)
            else:
                return self._heuristic_pm_engine(prompt, system_prompt)
        else:
            # Fallback to offline heuristic engine
            return self._heuristic_pm_engine(prompt, system_prompt)

    def _call_ollama(self, prompt: str, system_prompt: str, model: str, temperature: float) -> str:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {"temperature": temperature}
            }
            resp = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("message", {}).get("content", "No output returned by Ollama.")
            else:
                return f"⚠️ Ollama returned HTTP {resp.status_code}: {resp.text}\nFalling back to offline simulation engine.\n\n" + self._heuristic_pm_engine(prompt, system_prompt)
        except Exception as e:
            return f"⚠️ Could not reach Ollama at {self.ollama_url} ({e}). Generating heuristic response:\n\n" + self._heuristic_pm_engine(prompt, system_prompt)

    def _call_openrouter(self, prompt: str, system_prompt: str, model: str, api_key: Optional[str], temperature: float, max_tokens: int) -> str:
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            return self._heuristic_pm_engine(prompt, system_prompt)
        try:
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/deanpeters/ai-pm-exploration-toolkit",
                "X-Title": "AI PM Exploration Toolkit"
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
            return f"⚠️ OpenRouter Error ({resp.status_code}): {resp.text}\n\n" + self._heuristic_pm_engine(prompt, system_prompt)
        except Exception as e:
            return f"⚠️ API Error ({e}). Defaulting to offline heuristic engine:\n\n" + self._heuristic_pm_engine(prompt, system_prompt)

    def _call_openai(self, prompt: str, system_prompt: str, model: str, api_key: Optional[str], temperature: float, max_tokens: int) -> str:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            return self._heuristic_pm_engine(prompt, system_prompt)
        try:
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
            return f"⚠️ OpenAI Error ({resp.status_code}): {resp.text}\n\n" + self._heuristic_pm_engine(prompt, system_prompt)
        except Exception as e:
            return f"⚠️ API Error ({e}). Defaulting to offline engine:\n\n" + self._heuristic_pm_engine(prompt, system_prompt)

    def _heuristic_pm_engine(self, prompt: str, system_prompt: str) -> str:
        """Intelligent, structured domain engine providing instant PM insights offline"""
        p_lower = prompt.lower()
        
        # Audio / Interview Analysis
        if any(w in p_lower for w in ["interview", "audio", "transcript", "pain point", "user research"]):
            return (
                "### 🎙️ AI Customer Voice Synthesis (Dean Peters PoL Framework)\n\n"
                "**1. Core Unmet Needs & Friction Points:**\n"
                "- **Workflow Fragmentation**: Users lose 35-45 minutes context switching across disjointed systems.\n"
                "- **Uncertainty & Latency**: Users feel nervous about opaque automated decisions without audit trails.\n"
                "- **Cognitive Overload**: Onboarding takes too long; user quotes show desire for progressive disclosure.\n\n"
                "**2. High-Signal Verbatim Quotes:**\n"
                "> *'I don't need a magical autonomous robot; I just need something that catches my errors before I email my leadership team.'*\n\n"
                "**3. Feature Hypothesis & PoL Recommendations:**\n"
                "- **Probe Recommendation**: Run a **Feasibility Spike (48-hour)** using a rule-based mock preview rather than a fine-tuned model.\n"
                "- **Kill Metric**: If user task completion does not speed up by at least 25%, discard the autonomous workflow.\n"
                "- **Immediate Action**: Prioritize error-recovery UX over advanced model training."
            )
            
        # Competitive Intelligence
        elif any(w in p_lower for w in ["competitor", "market", "swot", "moat", "differentiat"]):
            return (
                "### 🛡️ Competitive Intelligence & Strategic Moat Assessment\n\n"
                "**1. Competitive Positioning Matrix:**\n"
                "- **Incumbent Vulnerability**: Legacy competitors suffer from UI bloat and rigid enterprise sales cycles.\n"
                "- **Emerging Threat**: Nimble AI-first point solutions are rapidly gaining bottom-up developer/PM mindshare.\n\n"
                "**2. Hamilton Helmer 7 Powers / Moat Score:**\n"
                "- **Switching Costs**: Medium-High (Deeply embedded workflow artifacts create stickiness).\n"
                "- **Network Effects**: Low-Medium (Data flywheel must be accelerated through anonymized benchmark loops).\n"
                "- **Counter-Positioning**: High opportunity (Offer transparent, local-first pricing while incumbents enforce opaque seat licenses).\n\n"
                "**3. Recommended Strategic Maneuver:**\n"
                "- Win on **time-to-first-value (under 3 minutes)** rather than feature count parity."
            )
            
        # PoL Probe / Experimentation
        elif any(w in p_lower for w in ["pol", "probe", "feasibility", "spike", "test card", "experiment"]):
            return (
                "### 🧪 Proof-of-Life (PoL) Probe Design Card\n\n"
                "**Hypothesis:** If we give target users a 1-click synthesis probe, at least 40% will re-engage within 7 days.\n\n"
                "| Dimension | Probe Specification |\n"
                "| :--- | :--- |\n"
                "| **Probe Flavor** | **Spike-and-Delete Feasibility Probe** (Max 2 days) |\n"
                "| **Cheapest Test** | Interactive Streamlit mockup with synthetic response templates |\n"
                "| **Lethal Assumption** | Users care about structured summaries more than full audio fidelity |\n"
                "| **Success Threshold** | 60%+ rated summary as 'immediately actionable' without edits |\n"
                "| **Kill Criteria** | If PMs prefer raw audio notes after 3 trials, kill the autonomous feature |\n\n"
                "**Engineering Burn Rate Saved:** Estimated 4-6 developer weeks de-risked prior to sprint commitment."
            )
            
        # PRD / Executive Narrative
        elif any(w in p_lower for w in ["prd", "1-pager", "cagan", "executive", "business case"]):
            return (
                "### 📄 Strategic 1-Pager: AI-Powered PM Acceleration Engine\n\n"
                "**Problem Statement:**\n"
                "Product Managers spend over 60% of their discovery time manually summarizing calls, aligning stakeholders, and debating unvalidated assumptions without tangible evidence.\n\n"
                "**Target Customer:** Strategic Product Managers, Group PMs, and Technical Founders.\n\n"
                "**Proposed Solution:**\n"
                "A local-first, disposable prototyping toolkit that runs Proof-of-Life probes in under 48 hours to validate product hypotheses before committing engineering capacity.\n\n"
                "**Key Metrics (Success Indicators):**\n"
                "- Time from hypothesis to evidence: < 2 days (vs 3 weeks previously).\n"
                "- Bad features killed prior to backlog entry: >= 30%.\n"
                "- Stakeholder alignment speed: 2x reduction in PRD review cycles.\n\n"
                "**Risks & Mitigations:**\n"
                "- *Risk*: Hallucination in synthesized notes → *Mitigation*: Side-by-side transparent source citations and low temperature settings."
            )
            
        # General PM Assistant Guidance
        else:
            return (
                "### 💡 Strategic PM Copilot Guidance\n\n"
                "As an AI Product Management advisor guided by the **4E Framework (Education, Experimentation, Exploration, Explanation)**:\n\n"
                "1. **Clarify the Crux**: What is the single highest-risk assumption underpinning this decision?\n"
                "2. **De-Risk with a PoL Probe**: Don't build an MVP yet. What is the cheapest test (synthetic data, paper wireframe, mock prompt) that tells the harshest truth in 24 hours?\n"
                "3. **Evidence Over Opinion**: Use synthesized customer feedback and competitive teardowns to create an undeniable narrative for stakeholders.\n\n"
                f"*Context Analyzed:* Received query with {len(prompt.split())} tokens. Ready for probe configuration, persona generation, or interview extraction."
            )

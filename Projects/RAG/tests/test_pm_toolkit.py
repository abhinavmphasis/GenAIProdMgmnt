"""
Verification test suite for pm_toolkit modules
"""

import sys
from pathlib import Path

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE_ROOT))

from pm_toolkit.llm_engine import LLMEngine
from pm_toolkit.pol_probes import PoLProbeStudio
from pm_toolkit.synthetic_personas import SyntheticPersonaGenerator
from pm_toolkit.audio_intelligence import AudioIntelligenceLab
from pm_toolkit.competitive_intel import CompetitiveIntelligenceLab
from pm_toolkit.prompt_evaluator import PromptEvaluator
from pm_toolkit.executive_narrative import ExecutiveNarrativeCrafter
from pm_toolkit.learning_classroom import LearningClassroom

def test_toolkit():
    print("Testing LLMEngine...")
    engine = LLMEngine()
    providers = engine.detect_providers()
    assert "offline_simulation" in providers
    response = engine.generate("Tell me about a feasibility spike", provider="heuristic-offline")
    assert len(response) > 50
    print("[OK] LLMEngine passed.")

    print("Testing PoLProbeStudio...")
    studio = PoLProbeStudio()
    score_res = studio.calculate_priority_score(8, 9, 2, 8)
    assert score_res["score"] > 60
    card = studio.generate_test_card("AI Feature", "Hypothesis 1", "feasibility_check", "Lethal assumption", "Pass if 80%", "Kill if 20%")
    assert "PROOF-OF-LIFE" in card
    print("[OK] PoLProbeStudio passed.")

    print("Testing SyntheticPersonaGenerator...")
    gen = SyntheticPersonaGenerator()
    personas = gen.generate_personas("b2b_saas", count=3)
    assert len(personas) == 3
    sim_res = gen.run_wind_tunnel_simulation("Automate sprint reporting with AI", personas)
    assert "acceptance_rate_pct" in sim_res
    df = gen.personas_to_dataframe(personas)
    assert len(df) == 3
    print("[OK] SyntheticPersonaGenerator passed.")

    print("Testing AudioIntelligenceLab...")
    audio_lab = AudioIntelligenceLab()
    proc_res = audio_lab.process_transcript("user_interview", audio_lab.sample_transcripts["user_interview"])
    assert len(proc_res["extracted_sections"]) > 0
    print("[OK] AudioIntelligenceLab passed.")

    print("Testing CompetitiveIntelligenceLab...")
    comp_lab = CompetitiveIntelligenceLab()
    moat_res = comp_lab.analyze_moat_and_vulnerabilities("OurApp", "IncumbentCorp")
    assert "moat_scores" in moat_res
    print("[OK] CompetitiveIntelligenceLab passed.")

    print("Testing PromptEvaluator...")
    pe = PromptEvaluator()
    eval_res = pe.evaluate_prompt("You are a helpful AI assistant. Only use provided context.")
    assert "clarity_score" in eval_res["scores"]
    print("[OK] PromptEvaluator passed.")

    print("Testing ExecutiveNarrativeCrafter...")
    crafter = ExecutiveNarrativeCrafter()
    one_pager = crafter.generate_cagan_one_pager("Auto-Sync", "Enterprise PMs", "Silos", "AI Sync", "30% time saved", "Low", "Low", "None", "85% liked probe")
    assert "PRODUCT OPPORTUNITY 1-PAGER" in one_pager
    print("[OK] ExecutiveNarrativeCrafter passed.")

    print("Testing LearningClassroom...")
    classroom = LearningClassroom()
    models = classroom.get_mental_models()
    assert len(models) >= 4
    print("[OK] LearningClassroom passed.")

    print("\nALL PM TOOLKIT MODULE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_toolkit()

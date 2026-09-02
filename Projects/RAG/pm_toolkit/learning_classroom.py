"""
The Personal AI Classroom & Learning Tracks
The 1st 'E' in Dean Peters' 4E Framework: Education.
Helps Product Managers transition from AI-Curious to AI-Confident
through mental models, evaluation cheat sheets, and fluency assessments.
"""

from typing import Dict, List, Any

class LearningClassroom:
    """Provides educational content, AI mental models, and interactive PM quizzes"""

    def get_4e_framework_breakdown(self) -> Dict[str, Any]:
        return {
            "title": "The Dean Peters 4E Learning Journey for PMs",
            "pillars": [
                {
                    "emoji": "🎓",
                    "name": "Education (Personal AI Classroom)",
                    "description": "Combat AI illiteracy. Build foundational intuition on prompt engineering, model trade-offs, and probabilistic systems in a safe sandbox.",
                    "action": "Understand tokens, hallucination vectors, and context management before writing PRDs."
                },
                {
                    "emoji": "🧪",
                    "name": "Experimentation (Evidence Over Opinion)",
                    "description": "Stop debating in committee meetings. Run disposable Proof-of-Life (PoL) Probes and synthetic customer simulations in 24-48 hours.",
                    "action": "Build spike-and-delete prototypes that cost zero engineering sprint points."
                },
                {
                    "emoji": "🔍",
                    "name": "Exploration (Discovery Without Limits)",
                    "description": "Discover latent opportunities across customer voice recordings, competitor vulnerabilities, and alternative AI models.",
                    "action": "Extract insights from customer calls and test edge cases before talking to real clients."
                },
                {
                    "emoji": "📊",
                    "name": "Explanation (Show Before Tell, Touch Before Sell)",
                    "description": "Transform stakeholder skepticism into leadership buy-in using tangible interactive evidence and Cagan 1-Pagers.",
                    "action": "Present live clickable probes with clear kill criteria to get budget approved."
                }
            ]
        }

    def get_mental_models(self) -> List[Dict[str, str]]:
        return [
            {
                "title": "1. Probabilistic vs. Deterministic Software",
                "summary": "Traditional software is deterministic (if user clicks X, always render Y). AI features are probabilistic (given input X, generate the statistically likely Y). As a PM, your job is no longer writing rigid acceptance tests, but designing confidence thresholds, graceful error degradation, and human-in-the-loop escape hatches."
            },
            {
                "title": "2. The 'Spike-and-Delete' PoL Probe",
                "summary": "A Proof-of-Concept (PoC) tests *how* to build something and often accidentally morphs into fragile production code. A Proof-of-Life (PoL) Probe is disposable reconnaissance designed to fail fast. If you don't delete your PoL probe code after gathering evidence, you did it wrong!"
            },
            {
                "title": "3. RAG vs. Fine-Tuning Decision Matrix",
                "summary": "Need your AI to know recent company data, private docs, or real-time facts? Use RAG (Retrieval-Augmented Generation). Need your AI to mimic a specific tone, dialect, specialized syntax, or reduce latency/token count? Fine-tune. 90% of PM problems require RAG, not fine-tuning."
            },
            {
                "title": "4. The Illusion of 100% Accuracy",
                "summary": "Never promise executives or users '100% accurate AI'. Treat AI as a brilliant, eager intern who needs supervision. Provide transparent source citations, highlight confidence scores, and let users edit results before submission."
            }
        ]

    def get_fluency_quiz_questions(self) -> List[Dict[str, Any]]:
        return [
            {
                "question": "An executive asks you to add an AI feature to auto-generate legal contracts. What is your first step as a modern AI PM?",
                "options": [
                    "Immediately create a Jira epic and assign 3 backend engineers to fine-tune a model.",
                    "Run a 48-hour PoL Probe with synthetic edge cases to determine hallucination risk and liability.",
                    "Wait 6 months for the legal team to draft complete specifications.",
                    "Pick the largest LLM on the market and integrate its API without testing."
                ],
                "correct_idx": 1,
                "explanation": "Correct! A 48-hour PoL Probe tests the lethal assumption (hallucination in legal clauses) before wasting developer resources."
            },
            {
                "question": "What is the primary difference between a PoC (Proof-of-Concept) and a PoL (Proof-of-Life) Probe?",
                "options": [
                    "A PoC is written in Python, while a PoL is written in Go.",
                    "A PoC tests engineering feasibility ('Can we build it?'), while a PoL tests value & lethal assumptions ('Does anyone care and is it fatal?').",
                    "There is no difference; they are identical buzzwords.",
                    "A PoC is free, whereas a PoL requires paid external consultants."
                ],
                "correct_idx": 1,
                "explanation": "Spot on. PoL Probes are lightweight, disposable reconnaissance tests to kill bad ideas early."
            },
            {
                "question": "Your company wants an AI assistant that answers questions from your internal 500-page policy handbook. Which architecture should you choose?",
                "options": [
                    "Fine-tune a Llama-3 model on the 500 pages.",
                    "Retrieval-Augmented Generation (RAG) with vector search and citations.",
                    "Train a foundation model from scratch on AWS.",
                    "Copy and paste the 500 pages into every user prompt."
                ],
                "correct_idx": 1,
                "explanation": "RAG is the gold standard for dynamic, factual reference retrieval with transparent citations."
            }
        ]

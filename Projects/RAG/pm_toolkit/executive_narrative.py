"""
Executive Narrative & Artifact Crafter
The 4th 'E' in Dean Peters' 4E Framework: Explanation.
Generates Marty Cagan 1-Pagers, Amazon-style Working Backwards PR/FAQs,
and Executive Business Cases based on findings from PoL Probes and customer research.
"""

from typing import Dict, Any, Optional

class ExecutiveNarrativeCrafter:
    """Generates C-suite and engineering alignment artifacts"""

    def generate_cagan_one_pager(
        self,
        product_name: str,
        target_customer: str,
        core_problem: str,
        proposed_solution: str,
        key_metrics: str,
        value_risk: str,
        feasibility_risk: str,
        viability_risk: str,
        pol_probe_evidence: str,
        author: str = "Abhinav"
    ) -> str:
        """Generates a Marty Cagan-style Product Opportunity 1-Pager"""
        return f"""# 📑 PRODUCT OPPORTUNITY 1-PAGER (Cagan Framework)
**Initiative:** {product_name}  
**Lead Product Manager:** {author}  
**Target Customer:** {target_customer}  
**Status:** Validated via Proof-of-Life (PoL) Reconnaissance Probe  

---

### 1. THE PROBLEM TO SOLVE
{core_problem}

### 2. PROPOSED SOLUTION & VALUE PROPOSITION
{proposed_solution}

### 3. EVIDENCE FROM PROOF-OF-LIFE (PoL) PROBE
> **Probe Finding:**  
> {pol_probe_evidence}

### 4. THE 4 BIG PRODUCT RISKS ASSESSMENT
| Product Risk | Risk Description & Mitigation |
| :--- | :--- |
| **Value Risk** (Will they buy/use it?) | {value_risk} |
| **Usability Risk** (Can they figure out how to use it?) | High initial cognitive friction mitigated by simplified zero-config UI and transparent citations. |
| **Feasibility Risk** (Can our engineers build it within cost?) | {feasibility_risk} |
| **Business Viability Risk** (Does it align with legal, compliance, ethics?) | {viability_risk} |

### 5. TARGET BUSINESS OUTCOMES (Success Metrics)
{key_metrics}

### 6. RECOMMENDATION
- [x] **PROCEED TO SPRINT DISCOVERY**: Lethal assumptions neutralized by PoL probe.
- [ ] **SPIKE-AND-DELETE (KILL)**: Hypotheses failed kill-criteria.
"""

    def generate_working_backwards_prfaq(
        self,
        headline: str,
        customer_name: str,
        customer_company: str,
        problem_summary: str,
        solution_summary: str,
        customer_quote: str,
        executive_quote: str,
        author: str = "Abhinav"
    ) -> str:
        """Generates an Amazon-style Working Backwards Press Release (PR/FAQ)"""
        return f"""# 📰 WORKING BACKWARDS PRESS RELEASE (Amazon PR/FAQ)

### FOR IMMEDIATE RELEASE

## {headline}

**SAN FRANCISCO, CA** — Today, the product team led by **{author}** unveiled an all-new capability designed to eradicate {problem_summary.lower()}.

For years, product and engineering leaders have struggled with unvalidated roadmap commitments and bloated development sprints. The new release solves this by {solution_summary.lower()}.

> "{customer_quote}"  
> — **{customer_name}**, {customer_company}

### Executive Perspective
> "{executive_quote}"  
> — **{author}, Head of Product**

### How Customers Get Started
Customers can activate the feature within 60 seconds directly from their existing workspace dashboard with zero configuration required.

---

### INTERNAL FAQ (Hard Questions Addressed)
**Q1: What was the lethal assumption we tested before writing code?**  
*A:* We conducted a 48-hour Proof-of-Life (PoL) probe with synthetic personas to confirm that users prioritized execution speed over complex customizations.

**Q2: What is the kill criteria if this underperforms post-launch?**  
*A:* If 30-day repeat engagement is under 35%, we will gracefully deprecate the feature without residual infrastructure drag.
"""

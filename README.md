# ProofGate

### From “we tested it” to “we proved it.”

**ProofGate** is a B2B AI robustness verification platform designed for companies building AI systems in safety-critical environments.

Traditional testing checks how a model behaves on a collection of examples. ProofGate takes a different approach: **formal verification**. Instead of testing only selected inputs, we define a mathematically bounded set of allowed changes and check whether the model's prediction remains unchanged throughout that set.

## The problem

AI is increasingly used in medicine, transportation, manufacturing, and other high-stakes fields. But even small input changes—such as shifts in image brightness or contrast—can sometimes cause a model to make a different prediction.

Testing thousands of examples can reveal failures, but passing those tests does not guarantee that no failure exists between the tested cases.

## Our solution

ProofGate helps teams evaluate and document model robustness through a simple workflow:

1. **Select a model and an input.** Choose the prediction to verify.
2. **Define allowed perturbations.** Set bounds for changes such as brightness and contrast.
3. **Run verification.** Analyze whether any input within those bounds can change the prediction.
4. **Review the result.** Receive a verification result, a counterexample when one is found, or an inconclusive result if a proof cannot be established.
5. **Generate a report.** Document the verification scope, conditions, outcome, and any discovered failures.

A successful proof applies **only to the specified model, input region, and assumptions**—not automatically to every possible real-world scenario.

## Demo: Pneumonia detection

Our hackathon demonstration compares two illustrative AI models performing the same task: classifying chest X-ray images for signs of pneumonia.

We vary two image properties:

- **Brightness** — how light or dark an image appears.
- **Contrast** — the difference between lighter and darker areas.

The demo shows how one model can change its prediction under a small adjustment, while another remains consistent over the demonstrated range. ProofGate aims to go beyond showing a few examples by verifying the **entire defined perturbation range** where supported.

> **Note:** This is a proof-of-concept demonstration, not a clinically validated diagnostic product. Robustness of a prediction is not the same as medical accuracy.

## Why formal verification?

| Conventional testing | ProofGate's approach |
| --- | --- |
| Evaluates sampled inputs | Analyzes a mathematically defined input set |
| Can discover failures | Can prove a bounded property or find a counterexample |
| Passing tests leaves untested cases | A successful proof covers the specified region |
| Often repeated manually | Designed for repeatable verification and reporting |

## Who is ProofGate for?

ProofGate is designed for **businesses developing or deploying high-stakes AI**, including teams working in:

- Healthcare and medical imaging
- Automotive and intelligent transportation
- Industrial automation and quality control
- Aviation and other safety-critical systems

Our proposed business model is an **annual B2B subscription**, with the long-term goal of making robustness checks part of the model development and update lifecycle.

## Regulatory context

**Article 15 of the AI Act** requires high-risk AI systems to meet appropriate standards of accuracy, robustness, and cybersecurity throughout their lifecycle. The relevant high-risk requirements have phased application dates, including **2 December 2027** for systems covered by Annex III and **2 August 2028** for systems covered by Annex I.

The AI Act **does not universally mandate formal verification**. ProofGate is intended to provide stronger technical evidence that can support robustness assessment and documentation; using ProofGate alone does not establish legal compliance.

- [AI Act — Article 15](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-15)
- [AI Act — implementation timeline](https://ai-act-service-desk.ec.europa.eu/en/ai-act/eu-ai-act-implementation-timeline)

## Project status

**Hackathon proof of concept.** The current focus is demonstrating the difference between empirical robustness testing and formal verification using a concrete, visual example.

Production integrations, supported model architectures, verification guarantees, and deployment options would need to be specified and validated before commercial use.

---

**ProofGate — From “we tested it” to “we proved it.”**

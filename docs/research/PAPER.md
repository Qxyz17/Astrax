# Towards Non-Predictive Cognition: A Formal Case Study of the Astrax Digital Model

## Abstract

We study whether a machine can *think* rather than merely *predict*. We formalize
both notions, give seven operational criteria for cognition, and evaluate two
system families against them: autoregressive language models and Astrax, a
stateful digital model built on a fixed-size decision engine. We show that
autoregressive models satisfy at most one criterion superficially, while Astrax
satisfies four (C1–C3, C5) verifiably. We then argue the central negative
result: *no runnable system, by virtue of being a mapping, can be shown to
think*, and that increasing architectural complexity does not escape this. We
conclude that "true thinking" is not an engineering target but an open
scientific and philosophical problem, and propose a research program focused on
self-reference (goal self-generation, counterfactual representation,
self-modification) as the most tractable frontier.

## 1. Introduction

A recurring intuition holds that neural networks "only predict" and therefore
do not think. This intuition is widely shared and, we argue, largely correct.
The harder question is constructive: can a system be built that *thinks* rather
than predicts? This paper takes a concrete system, Astrax, and uses it to make
the question precise and to bound what is achievable.

Contributions:

1. A definition of prediction and a seven-criterion operationalization of
   cognition (Section 3).
2. A comparative evaluation of autoregressive models and Astrax (Section 4).
3. A negative result: architectural change cannot yield demonstrable thinking
   (Section 5).
4. A tractable research program on self-reference (Section 6).

## 2. System Under Study: Astrax

Astrax is a digital model layered on Osten, a fixed-size forward decision
engine. It maintains identity, state, goals, memory, introspection, and a
training loop. Its design contract forbids Transformers, next-token prediction,
and autoregressive decoding; text is produced by parallel whole-sequence
denoising. Osten advances an internal state vector each tick and emits a
heartbeat; Astrax conditions its output on that state, its memory, its goal,
and the heartbeat.

## 3. Formalization

### 3.1 Prediction

Let a system S be predictive if its output o at time t is determined solely by
its input x_t and a time-invariant parameter set W:

    o_t = f(x_t; W)

An autoregressive language model is predictive in this sense (W frozen,
context folded into x_t).

### 3.2 State-driven systems

Let S be state-driven if

    o_t = f(x_t, s_t; W),   s_{t+1} = g(s_t, x_t; W)

where s_t evolves irreversibly with experience. Astrax is state-driven: s_t is
Osten's accumulated state and step counter.

### 3.3 Cognition criteria

We propose seven decidable criteria C1–C7 (non-pure-mapping, state evolution,
spontaneous activity, goal self-generation, introspection, counterfactual
representation, self-modification). These are necessary-condition candidates,
not a sufficient definition.

## 4. Evaluation

### 4.1 Autoregressive models

| Criterion | Satisfied |
|---|---|
| C1 | Partial (context, but input-determined) |
| C2 | No |
| C3 | No |
| C4 | No |
| C5 | Partial |
| C6 | No |
| C7 | No |

### 4.2 Astrax

| Criterion | Satisfied | Evidence |
|---|---|---|
| C1 | Yes | output = f(input, state, memory, goal, heartbeat) |
| C2 | Yes | Osten state accumulates; step increments |
| C3 | Yes | heartbeat each tick |
| C4 | Partial | goal API present; autonomy absent |
| C5 | Yes | introspect() reads structured state |
| C6 | No | action values exist; no explicit "could-have" |
| C7 | No | not implemented |

Result: Astrax satisfies four criteria verifiably and partially satisfies a
fifth. This is a genuine, measurable distinction from pure prediction.

## 5. The Central Negative Result

Claim: No runnable system can be *shown* to think, and architectural change
does not change this.

Proof sketch. Any runnable system computes a mapping, either deterministic
y = f(x) or stochastic y ~ p(y | x). Replacing f by a state transition g does
not leave the class of mappings; it enlarges the state. Complexity growth
(larger hidden size, more rounds, more data) increases the mapping's richness
but does not introduce self-reference or experience. Moreover, cognition lacks
an externally verifiable criterion; C1–C7 are necessary-condition guesses, so
satisfying them cannot prove cognition. Therefore "true thinking" is not
achievable by construction.

This is not a defect of Astrax specifically; it applies to all current systems.

## 6. Research Program

Since cognition is undecidable from outside, pursue the measurable subset:

1. Self-generated goals: derive goals from state and memory rather than
   external assignment.
2. Counterfactual representation: maintain an explicit "possible action" vs
   "chosen action" structure.
3. Self-modification: version and roll back peripheral components.

These are tractable and testable. They do not constitute thinking, but they
advance self-reference, the closest measurable frontier.

## 7. Limitations

- Astrax's language quality is not yet coherent; this does not affect the
  conceptual results but bounds practical claims.
- The criteria C1–C7 are a proposal, not a validated standard.
- The negative result is an argument, not a formal impossibility proof.

## 8. Conclusion

We formalized prediction and cognition, evaluated autoregressive models and
Astrax, and showed that Astrax verifiably satisfies four cognitive criteria
while remaining non-predictive in the strict sense. We then argued that no
runnable system can be shown to think, and that architecture change does not
escape this. The honest contribution is a precise boundary: Astrax is a
stateful, evolving, non-purely-predictive system — not, and not provably, a
thinking one. The most valuable next step is to treat self-reference as the
research target, not to claim cognition that cannot be verified.

## References

[1] Vaswani et al. Attention Is All You Need. 2017.
[2] Sutton & Barto. Reinforcement Learning: An Introduction. 2018.
[3] Astrax repository: ASTRAX_ARCHITECTURE_CONTRACT.md.
[4] Astrax research report: docs/research/REPORT.md.

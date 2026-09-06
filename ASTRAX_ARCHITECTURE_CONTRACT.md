# Astrax Architecture Contract

This file is a repository-level, non-negotiable design contract.

## Permanent rules

1. Astrax is **not a Transformer**.
2. Astrax does **not** use next-token prediction or autoregressive token-by-token generation.
3. Osten remains the fixed-size core forward decision engine.
4. Astrax owns identity, state, goals, memory, introspection, interaction, training, and presentation.
5. The final decision must come from the learned Osten/Astrax state-transition and value path. Keywords, hand-written rules, and direct retrieval must not select the answer.
6. Text conversation is holistic: the complete input is encoded first, Osten advances the internal state, and a complete response representation is decoded in parallel response slots.
7. Text and code are outputs of the learned state/action/response system, not token-probability continuation.
8. Offline RL data must use real transitions:

   ```text
   state, action, reward, next_state, terminal
   ```

9. The core implementation is C++17 with native MSVC Visual Studio projects. CMake is not part of Astrax.
10. Future modalities must enter through the same structured state/action interface without changing the non-Transformer, non-next-token contract.

## Implementation guard

The same rules are represented in `include/astrax/architecture.hpp` and checked with
compile-time assertions. Changing these constants requires an explicit architecture
review and corresponding test updates.

The current dialogue implementation is a holistic response-slot decoder:

```text
complete input text
        -> character n-gram/vector encoder
        -> Osten state/action decision
        -> parallel fixed response-slot classifiers
        -> complete response text
```

No decoder slot reads a previously emitted slot. The implementation is deliberately
not a Transformer and is deliberately not next-token training.

# Rules for working on this repo

## Code style
- Keep code simple and readable, for each function add a comment detailing what it does. Be concise, explain inputs and outpus of functions precisely (ie. type and size). Prefer plain numpy/OpenCV over extra libraries.
- Every module starts with a docstring explaining what it does and how it's used.
- Comment the *why*
- Use units in variable names where relevant: `_px` for pixels, `_cm` for centimetres, `_s` for seconds.
- Missing data is NaN, never a made-up value.
- Heavy imports (mediapipe, torch, libero) go inside the functions that need them.

## Structure
- Logic lives in `src/hand2panda/`; `scripts/` only chain modules together; notebooks only call scripts.
- Don't add new top-level folders or dependencies without asking me first.

## Working with me
- I'm not a CS specialist: explain every new piece of code in plain language, step by step.
- Show me code and wait for my OK before committing or pushing anything.
- I run and test the code myself in Colab; don't claim something works unless I've confirmed it.
- Don't write marketing-style text in the README. Be factual and honest about what didn't work.
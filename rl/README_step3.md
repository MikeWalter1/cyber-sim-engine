# RL Step 3: Observation Encoding and Action Masks

This step adds the first model-facing interface:

- `rl/observation_encoder.js` converts `{ board, waitingFor }` into a fixed-size numeric vector.
- `rl/action_space.js` converts legal engine actions into a fixed-size action mask.
- `rl/smoke_test_observations.js` stress-tests observations and masks while random agents play full games.

This package assumes Step 2 is already installed:

- `rl/env_runner.js`
- `rl/legal_actions.js`
- the `lib/select.js` hotfix from Step 2, if your local engine still needs it

## Install

Extract the ZIP into the `cyber-sim-engine-main` repo root so the files land here:

```text
cyber-sim-engine-main/
  rl/
    action_space.js
    observation_encoder.js
    smoke_test_observations.js
    README_step3.md
```

## Run

From the repo root:

```bash
node rl/smoke_test_observations.js --deck1 decks/DEV-TEST-001.deck --deck2 decks/DEV-TEST-002.deck --games 100 --seed 42
```

Stronger test:

```bash
node rl/smoke_test_observations.js --deck1 decks/DEV-TEST-001.deck --deck2 decks/DEV-TEST-002.deck --games 1000 --seed 42
```

Optional slower validation mode:

```bash
node rl/smoke_test_observations.js --deck1 decks/DEV-TEST-001.deck --deck2 decks/DEV-TEST-002.deck --games 100 --seed 42 --prevalidate
```

## Success target

The important fields are:

```json
{
  "ok": true,
  "errors": 0,
  "illegalActions": 0,
  "badObservations": 0,
  "badActionMasks": 0,
  "actionMaskOverflows": 0
}
```

## What the observation contains

The encoder uses a fixed-size vector with:

- global turn, phase, current player, and waiting-step features;
- effect-choice kind features;
- decision metadata such as available dice, attackable count, blocker count, and choice ranges;
- self/opponent player summaries;
- padded self/opponent gig lists;
- padded self/opponent field unit lists;
- padded self hand card list.

The current observation is intentionally simple. It is good enough to test RL plumbing, but not yet meant to produce a strong agent.

## How action masks work

The model should output an integer action index. The action index selects from the current state's `legalActions` array.

`actionMask[i] = 1` means index `i` is legal.

Default fixed action dimension:

```js
DEFAULT_MAX_ACTIONS = 128
```

The Step 2 random test observed a max of 47 generated actions with the dev decks, so 128 leaves room for growth. If `actionMaskOverflows > 0`, increase the size:

```bash
node rl/smoke_test_observations.js --deck1 decks/DEV-TEST-001.deck --deck2 decks/DEV-TEST-002.deck --games 1000 --seed 42 --max-actions 256
```

## Next step

The next milestone is a Python or JS Gym-style wrapper that returns:

```js
{
  observation: observation.vector,
  actionMask,
  legalActions,
  reward,
  terminated,
  truncated,
  info
}
```

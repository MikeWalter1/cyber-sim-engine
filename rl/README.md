# RL Step 2: Legal actions and random-agent smoke test

This folder contains the second RL milestone for `cyber-sim-engine`.

It adds a legal-action generator and a random-agent stress test. The goal is not strong play yet. The goal is to prove that agents can ask the environment for concrete legal actions and play many complete games without crashing the engine.

## Files

- `env_runner.js`  
  Same minimal environment wrapper from Step 1.

- `legal_actions.js`  
  Generates concrete engine actions from `{ board, waitingFor }`.

- `smoke_test_random_agent.js`  
  Runs random-vs-random games using generated legal actions.

## Install location

Extract this ZIP directly into the `cyber-sim-engine` repo root.

Expected result:

```text
cyber-sim-engine/
  rl/
    env_runner.js
    legal_actions.js
    smoke_test_random_agent.js
    README.md
```

## Run a small test

From the repo root:

```bash
node rl/smoke_test_random_agent.js \
  --deck1 decks/DEV-TEST-001.deck \
  --deck2 decks/DEV-TEST-002.deck \
  --games 100 \
  --seed 42
```

## Run a stronger test

```bash
node rl/smoke_test_random_agent.js \
  --deck1 decks/DEV-TEST-001.deck \
  --deck2 decks/DEV-TEST-002.deck \
  --games 1000 \
  --seed 42
```

## Optional candidate prevalidation

This asks the engine to trial-step every generated candidate action before selecting one. It is slower, but useful while debugging the legal-action generator.

```bash
node rl/smoke_test_random_agent.js \
  --deck1 decks/DEV-TEST-001.deck \
  --deck2 decks/DEV-TEST-002.deck \
  --games 100 \
  --seed 42 \
  --prevalidate
```

## Expected success condition

The important fields are:

```json
{
  "ok": true,
  "errors": 0,
  "illegalActions": 0
}
```

Win rates do not matter yet. Random play is only a stability test.

## Current legal-action coverage

Covered now:

- `choose_gig_die`
- `main_phase`
  - `end_turn`
  - `tap_resource`
  - `untap_resource`
  - `sell_card`
  - `call_legend`
  - affordable `play_card`
  - basic Gear equip targets
  - `declare_attack`
  - `activate_anytime_spend`
- `attacker_interrupt_step`
  - pass
  - spend-activated interrupt assets
- `defensive_step`
  - pass
  - block
  - defensive legend call
  - spend-activated interrupt assets
- `choose_gig_to_steal`
- `effect_choice`

Intentionally left for a later milestone:

- interrupt card casting with `play_card_interrupt_cast`
- advanced action ranking
- observation tensors
- Python/Gymnasium bridge

## Use from another script

```js
const { CyberSimEnv } = require('./rl/env_runner');
const { getLegalActions, makeActionMask } = require('./rl/legal_actions');

const env = new CyberSimEnv({
  deck1Path: 'decks/DEV-TEST-001.deck',
  deck2Path: 'decks/DEV-TEST-002.deck',
  seed: 42,
});

let state = env.reset();

const legalActions = getLegalActions(env.board, env.waitingFor, {
  db: env.db,
  scripts: env.scripts,
  evalExpr: env.engine.evalExpr,
});

const actionMask = makeActionMask(legalActions);

state = env.step(legalActions[0]);
```

## Next milestone

Add an observation encoder:

```js
encodeObservation(board, waitingFor, pid)
```

That will turn the board state into a compact vector or JSON observation suitable for a Python/Gymnasium wrapper.

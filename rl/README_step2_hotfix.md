# Step 2 Hotfix: `from_event` Owner Inference

This hotfix patches an engine edge case found by the 1000-game random-agent test.

## Problem

A random game reached `choose_gig_to_steal` and selected a gig from `waitingFor.available_iids`, but the engine crashed with:

```text
Cannot read properties of undefined (reading 'zones')
```

The selected action was structurally valid. The crash can happen after a steal when a card effect uses `target.from_event: "stolen_gigs"`. The stolen gig objects in the event payload may not carry `_pid`/`pid`, so later effect resolution can try to access `b[undefined].zones`.

## Fix

`lib/select.js` now infers the owner pid for event-provided refs by scanning the current board. For stolen gigs, the gig has already been moved to the stealing player's `zones.gigs`, so the pid can be recovered safely.

## Install

Extract this ZIP into the repo root and overwrite:

```text
lib/select.js
```

## Re-test

```bash
node rl/smoke_test_random_agent.js --deck1 decks/DEV-TEST-001.deck --deck2 decks/DEV-TEST-002.deck --games 1000 --seed 42
```

Expected target:

```json
{
  "ok": true,
  "errors": 0,
  "illegalActions": 0
}
```

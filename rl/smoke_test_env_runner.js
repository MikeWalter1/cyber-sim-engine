#!/usr/bin/env node
'use strict';

/**
 * Smoke test for rl/env_runner.js.
 *
 * Run from the cyber-sim-engine repo root:
 *
 *   node rl/smoke_test_env_runner.js \
 *     --deck1 decks/DEV-TEST-001.deck \
 *     --deck2 decks/DEV-TEST-002.deck \
 *     --seed 42
 *
 * This is not an intelligent agent.
 * It only proves the reset/step wrapper works by using safe actions:
 *   - choose the first available gig die
 *   - use engine default passes when available
 */

const path = require('path');
const { CyberSimEnv } = require('./env_runner');

function parseArgs(argv) {
  const out = {
    engineRoot: path.resolve(__dirname, '..'),
    deck1Path: null,
    deck2Path: null,
    firstPlayer: null,
    seed: undefined,
    turnCap: 200,
    maxSteps: 1000,
    printSteps: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    const next = argv[i + 1];

    if (key === '--engine-root' && next) {
      out.engineRoot = path.resolve(next);
      i++;
    } else if (key === '--deck1' && next) {
      out.deck1Path = path.resolve(next);
      i++;
    } else if (key === '--deck2' && next) {
      out.deck2Path = path.resolve(next);
      i++;
    } else if (key === '--first' && next) {
      out.firstPlayer = next;
      i++;
    } else if (key === '--seed' && next) {
      out.seed = parseInt(next, 10);
      i++;
    } else if (key === '--turn-cap' && next) {
      out.turnCap = parseInt(next, 10);
      i++;
    } else if (key === '--max-steps' && next) {
      out.maxSteps = parseInt(next, 10);
      i++;
    } else if (key === '--print-steps') {
      out.printSteps = true;
    }
  }

  if (!out.deck1Path) {
    out.deck1Path = path.join(out.engineRoot, 'decks', 'DEV-TEST-001.deck');
  }

  if (!out.deck2Path) {
    out.deck2Path = path.join(out.engineRoot, 'decks', 'DEV-TEST-002.deck');
  }

  return out;
}

function firstOrNull(array) {
  return Array.isArray(array) && array.length > 0 ? array[0] : null;
}

function selectSmokeTestAction(env, state) {
  const waitingFor = state.waitingFor;

  if (!waitingFor) {
    return null;
  }

  const pass = env.defaultPassAction(waitingFor);

  if (pass) {
    return pass;
  }

  switch (waitingFor.step) {
    case 'choose_gig_die': {
      const sides = firstOrNull(waitingFor.available);
      if (!sides) {
        throw new Error('choose_gig_die has no available die sides');
      }
      return { step: 'choose_gig_die', sides };
    }

    case 'choose_gig_to_steal': {
      const iids = (waitingFor.available_iids || []).slice(0, waitingFor.count || 0);
      return { step: 'choose_gig_to_steal', iids };
    }

    case 'effect_choice': {
      const choice = waitingFor.choice_needed || {};

      switch (choice.kind) {
        case 'confirm_optional':
          return { step: 'effect_choice_response', response: { accept: false } };

        case 'choose_amount':
          return { step: 'effect_choice_response', response: { amount: choice.min || 0 } };

        case 'choose_unit':
        case 'choose_gig':
        case 'choose_legend':
        case 'choose_gear':
        case 'choose_card_in_hand':
        case 'choose_card_in_trash':
        case 'choose_card_in_deck': {
          const available = choice.available_iids || [];
          const iid = available.length > 0 ? available[0] : null;
          return { step: 'effect_choice_response', response: { iid } };
        }

        case 'choose_from_top_n': {
          return { step: 'effect_choice_response', response: { selected_iids: [] } };
        }

        case 'choose_units': {
          return { step: 'effect_choice_response', response: { selected_iids: [] } };
        }

        default:
          throw new Error(`Smoke test does not know how to answer effect_choice kind: ${choice.kind}`);
      }
    }

    default:
      throw new Error(`Smoke test does not know how to answer waitingFor.step: ${waitingFor.step}`);
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  const env = new CyberSimEnv({
    engineRoot: args.engineRoot,
    deck1Path: args.deck1Path,
    deck2Path: args.deck2Path,
    firstPlayer: args.firstPlayer,
    seed: args.seed,
    turnCap: args.turnCap,
  });

  let state = env.reset({
    seed: args.seed,
    firstPlayer: args.firstPlayer,
  });

  if (args.printSteps) {
    console.log(JSON.stringify({
      event: 'reset',
      turn: state.turn,
      phase: state.phase,
      currentPlayer: state.currentPlayer,
      waitingFor: state.waitingFor,
    }, null, 2));
  }

  while (!state.done && state.steps < args.maxSteps) {
    const action = selectSmokeTestAction(env, state);

    if (!action) {
      break;
    }

    state = env.step(action);

    if (args.printSteps) {
      console.log(JSON.stringify({
        event: 'step',
        steps: state.steps,
        action,
        turn: state.turn,
        phase: state.phase,
        currentPlayer: state.currentPlayer,
        waitingForStep: state.waitingFor ? state.waitingFor.step : null,
        winner: state.winner,
        done: state.done,
        truncated: state.truncated,
      }, null, 2));
    }
  }

  const summary = {
    ok: !state.error,
    winner: state.winner,
    done: state.done,
    terminated: state.terminated,
    truncated: state.truncated,
    turn: state.turn,
    steps: state.steps,
    phase: state.phase,
    currentPlayer: state.currentPlayer,
    waitingForStep: state.waitingFor ? state.waitingFor.step : null,
  };

  console.log(JSON.stringify(summary, null, 2));
}

main();

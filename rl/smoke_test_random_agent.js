#!/usr/bin/env node
'use strict';

/**
 * Random legal-action smoke test for cyber-sim-engine.
 *
 * Run from the cyber-sim-engine repo root:
 *
 *   node rl/smoke_test_random_agent.js \
 *     --deck1 decks/DEV-TEST-001.deck \
 *     --deck2 decks/DEV-TEST-002.deck \
 *     --games 100 \
 *     --seed 42
 */

const path = require('path');
const { CyberSimEnv } = require('./env_runner');
const {
  getLegalActions,
  filterActionsByTrial,
  cloneJson,
} = require('./legal_actions');

function parseArgs(argv) {
  const out = {
    engineRoot: path.resolve(__dirname, '..'),
    deck1Path: null,
    deck2Path: null,
    firstPlayer: null,
    seed: 1,
    games: 100,
    turnCap: 200,
    maxStepsPerGame: 5000,
    maxMainActionsPerTurn: 12,
    prevalidate: false,
    verbose: false,
    printFirstError: true,
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
    } else if (key === '--games' && next) {
      out.games = parseInt(next, 10);
      i++;
    } else if (key === '--turn-cap' && next) {
      out.turnCap = parseInt(next, 10);
      i++;
    } else if (key === '--max-steps-per-game' && next) {
      out.maxStepsPerGame = parseInt(next, 10);
      i++;
    } else if (key === '--max-main-actions-per-turn' && next) {
      out.maxMainActionsPerTurn = parseInt(next, 10);
      i++;
    } else if (key === '--prevalidate') {
      out.prevalidate = true;
    } else if (key === '--verbose') {
      out.verbose = true;
    } else if (key === '--no-first-error') {
      out.printFirstError = false;
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

function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function chooseRandom(array, rng) {
  if (!Array.isArray(array) || array.length === 0) {
    return null;
  }
  return array[Math.floor(rng() * array.length)];
}

function shouldForceEndTurn(state, mainActionCounts, maxMainActionsPerTurn) {
  if (!state || !state.waitingFor || state.waitingFor.step !== 'main_phase') {
    return false;
  }

  const key = `${state.turn}:${state.currentPlayer}`;
  const count = mainActionCounts.get(key) || 0;
  return count >= maxMainActionsPerTurn;
}

function rememberMainAction(state, action, mainActionCounts) {
  if (!state || !state.waitingFor || state.waitingFor.step !== 'main_phase') {
    return;
  }

  if (action && action.step === 'end_turn') {
    return;
  }

  const key = `${state.turn}:${state.currentPlayer}`;
  mainActionCounts.set(key, (mainActionCounts.get(key) || 0) + 1);
}

function selectAction(env, state, rng, options, counters) {
  let actions = getLegalActions(env.board, env.waitingFor, {
    db: env.db,
    scripts: env.scripts,
    evalExpr: env.engine.evalExpr,
  });

  counters.decisions++;
  counters.totalGeneratedActions += actions.length;
  counters.maxGeneratedActions = Math.max(counters.maxGeneratedActions, actions.length);

  if (options.prevalidate) {
    const filtered = filterActionsByTrial(env.engine, env.board, actions, env.db, env.scripts);
    counters.rejectedCandidateActions += filtered.rejected.length;
    if (filtered.rejected.length > 0 && !counters.firstRejectedCandidate) {
      counters.firstRejectedCandidate = {
        turn: state.turn,
        phase: state.phase,
        currentPlayer: state.currentPlayer,
        waitingForStep: state.waitingFor ? state.waitingFor.step : null,
        rejected: filtered.rejected[0],
      };
    }
    actions = filtered.valid;
  }

  if (actions.length === 0) {
    const fallback = env.defaultPassAction(env.waitingFor);
    if (fallback) {
      counters.fallbackActions++;
      return fallback;
    }
    return null;
  }

  if (shouldForceEndTurn(state, counters.mainActionCounts, options.maxMainActionsPerTurn)) {
    const endTurn = actions.find((action) => action.step === 'end_turn');
    if (endTurn) {
      counters.forcedEndTurns++;
      return endTurn;
    }
  }

  // Bias slightly toward ending the turn so random play keeps moving.
  if (state.waitingFor && state.waitingFor.step === 'main_phase') {
    const endTurn = actions.find((action) => action.step === 'end_turn');
    if (endTurn && rng() < 0.20) {
      return endTurn;
    }
  }

  return chooseRandom(actions, rng);
}

function makeEnv(args) {
  return new CyberSimEnv({
    engineRoot: args.engineRoot,
    deck1Path: args.deck1Path,
    deck2Path: args.deck2Path,
    firstPlayer: args.firstPlayer,
    seed: args.seed,
    turnCap: args.turnCap,
  });
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const env = makeEnv(args);
  const rng = mulberry32(args.seed || 1);

  const stats = {
    ok: true,
    games: args.games,
    p1Wins: 0,
    p2Wins: 0,
    noWinner: 0,
    errors: 0,
    illegalActions: 0,
    truncated: 0,
    maxStepLimit: 0,
    totalTurns: 0,
    totalSteps: 0,
    decisions: 0,
    totalGeneratedActions: 0,
    maxGeneratedActions: 0,
    rejectedCandidateActions: 0,
    fallbackActions: 0,
    forcedEndTurns: 0,
    firstError: null,
    firstRejectedCandidate: null,
    errorBuckets: {},
  };

  for (let gameIndex = 0; gameIndex < args.games; gameIndex++) {
    const gameSeed = Number.isInteger(args.seed) ? args.seed + gameIndex : undefined;
    let state = env.reset({
      seed: gameSeed,
      firstPlayer: args.firstPlayer,
    });

    // High-throughput random testing does not need engine traces.
    if (env.board && typeof env.engine.disableTrace === 'function') {
      env.engine.disableTrace(env.board);
    }

    const counters = {
      decisions: 0,
      totalGeneratedActions: 0,
      maxGeneratedActions: 0,
      rejectedCandidateActions: 0,
      fallbackActions: 0,
      forcedEndTurns: 0,
      firstRejectedCandidate: null,
      mainActionCounts: new Map(),
    };

    let gameError = null;

    while (!state.done && state.steps < args.maxStepsPerGame) {
      const action = selectAction(env, state, rng, args, counters);

      if (!action) {
        gameError = new Error(`No legal action for waitingFor.step=${state.waitingFor ? state.waitingFor.step : 'null'}`);
        break;
      }

      rememberMainAction(state, action, counters.mainActionCounts);

      try {
        state = env.step(cloneJson(action));
      } catch (error) {
        gameError = error;
        stats.illegalActions++;

        if (!stats.firstError) {
          stats.firstError = {
            game: gameIndex + 1,
            seed: gameSeed,
            turn: state.turn,
            phase: state.phase,
            currentPlayer: state.currentPlayer,
            waitingForStep: state.waitingFor ? state.waitingFor.step : null,
            action,
            error: error.message,
          };
        }
        break;
      }
    }

    stats.decisions += counters.decisions;
    stats.totalGeneratedActions += counters.totalGeneratedActions;
    stats.maxGeneratedActions = Math.max(stats.maxGeneratedActions, counters.maxGeneratedActions);
    stats.rejectedCandidateActions += counters.rejectedCandidateActions;
    stats.fallbackActions += counters.fallbackActions;
    stats.forcedEndTurns += counters.forcedEndTurns;

    if (!stats.firstRejectedCandidate && counters.firstRejectedCandidate) {
      stats.firstRejectedCandidate = {
        game: gameIndex + 1,
        seed: gameSeed,
        ...counters.firstRejectedCandidate,
      };
    }

    if (gameError) {
      stats.errors++;
      const key = gameError.message || String(gameError);
      stats.errorBuckets[key] = (stats.errorBuckets[key] || 0) + 1;
    } else if (!state.done && state.steps >= args.maxStepsPerGame) {
      stats.maxStepLimit++;
      stats.noWinner++;
    } else if (state.truncated) {
      stats.truncated++;
      stats.noWinner++;
    } else if (state.winner === 'p1') {
      stats.p1Wins++;
    } else if (state.winner === 'p2') {
      stats.p2Wins++;
    } else {
      stats.noWinner++;
    }

    stats.totalTurns += state.turn || 0;
    stats.totalSteps += state.steps || 0;

    if (args.verbose) {
      process.stderr.write(`game ${gameIndex + 1}/${args.games}: winner=${state.winner || 'none'} turns=${state.turn} steps=${state.steps}${gameError ? ` error=${gameError.message}` : ''}\n`);
    }
  }

  stats.ok = stats.errors === 0 && stats.illegalActions === 0;
  stats.avgTurns = args.games > 0 ? Number((stats.totalTurns / args.games).toFixed(2)) : 0;
  stats.avgSteps = args.games > 0 ? Number((stats.totalSteps / args.games).toFixed(2)) : 0;
  stats.avgGeneratedActions = stats.decisions > 0
    ? Number((stats.totalGeneratedActions / stats.decisions).toFixed(2))
    : 0;

  if (!args.printFirstError) {
    delete stats.firstError;
    delete stats.firstRejectedCandidate;
  }

  console.log(JSON.stringify(stats, null, 2));

  if (!stats.ok) {
    process.exitCode = 1;
  }
}

main();

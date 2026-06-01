'use strict';

/**
 * Fixed-size action mask helper for cyber-sim-engine RL experiments.
 *
 * This file intentionally keeps the engine action objects unchanged. The model
 * sees an index into the current state's legalActions array, while actionMask
 * marks which indices are selectable inside a fixed maximum action dimension.
 */

const { getLegalActions } = require('./legal_actions');

const DEFAULT_MAX_ACTIONS = 128;

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function sumMask(mask) {
  return asArray(mask).reduce((sum, value) => sum + (value ? 1 : 0), 0);
}

function makeFixedActionMask(actionCount, maxActions = DEFAULT_MAX_ACTIONS) {
  if (!Number.isInteger(maxActions) || maxActions <= 0) {
    throw new Error(`maxActions must be a positive integer, got ${maxActions}`);
  }

  const safeActionCount = Math.max(0, Number.isInteger(actionCount) ? actionCount : 0);
  const mask = new Array(maxActions).fill(0);
  const n = Math.min(safeActionCount, maxActions);

  for (let i = 0; i < n; i++) {
    mask[i] = 1;
  }

  return mask;
}

function buildActionInterface(board, waitingFor, context = {}, options = {}) {
  const maxActions = options.maxActions == null ? DEFAULT_MAX_ACTIONS : options.maxActions;
  const legalActions = getLegalActions(board, waitingFor, context);
  const actionCount = legalActions.length;
  const actionMask = makeFixedActionMask(actionCount, maxActions);

  return {
    legalActions,
    actionMask,
    actionCount,
    maxActions,
    overflow: actionCount > maxActions,
    maskedActionCount: Math.min(actionCount, maxActions),
  };
}

function validateActionInterface(actionInterface) {
  const errors = [];

  if (!actionInterface || typeof actionInterface !== 'object') {
    return { ok: false, errors: ['actionInterface is not an object'] };
  }

  const legalActions = asArray(actionInterface.legalActions);
  const actionMask = actionInterface.actionMask;
  const maxActions = actionInterface.maxActions;
  const actionCount = actionInterface.actionCount;

  if (!Number.isInteger(maxActions) || maxActions <= 0) {
    errors.push(`invalid maxActions: ${maxActions}`);
  }

  if (!Number.isInteger(actionCount) || actionCount !== legalActions.length) {
    errors.push(`actionCount ${actionCount} does not match legalActions.length ${legalActions.length}`);
  }

  if (!Array.isArray(actionMask)) {
    errors.push('actionMask is not an array');
  } else {
    if (actionMask.length !== maxActions) {
      errors.push(`actionMask.length ${actionMask.length} does not equal maxActions ${maxActions}`);
    }

    for (let i = 0; i < actionMask.length; i++) {
      if (actionMask[i] !== 0 && actionMask[i] !== 1) {
        errors.push(`actionMask[${i}] is ${actionMask[i]}, expected 0 or 1`);
        break;
      }
    }

    const expectedOnes = Math.min(Math.max(0, actionCount || 0), Math.max(0, maxActions || 0));
    const actualOnes = sumMask(actionMask);
    if (actualOnes !== expectedOnes) {
      errors.push(`actionMask has ${actualOnes} active entries, expected ${expectedOnes}`);
    }

    for (let i = 0; i < actionMask.length; i++) {
      const expected = i < expectedOnes ? 1 : 0;
      if (actionMask[i] !== expected) {
        errors.push(`actionMask[${i}] is ${actionMask[i]}, expected contiguous mask value ${expected}`);
        break;
      }
    }
  }

  if (actionInterface.overflow !== (legalActions.length > maxActions)) {
    errors.push('overflow flag does not match legalActions.length > maxActions');
  }

  return { ok: errors.length === 0, errors };
}

function selectActionByIndex(actionInterface, actionIndex) {
  if (!actionInterface || !Array.isArray(actionInterface.legalActions)) {
    throw new Error('Invalid actionInterface');
  }

  if (!Number.isInteger(actionIndex)) {
    throw new Error(`actionIndex must be an integer, got ${actionIndex}`);
  }

  if (actionIndex < 0 || actionIndex >= actionInterface.maxActions) {
    throw new Error(`actionIndex ${actionIndex} outside fixed action space [0, ${actionInterface.maxActions})`);
  }

  if (!actionInterface.actionMask || actionInterface.actionMask[actionIndex] !== 1) {
    throw new Error(`actionIndex ${actionIndex} is masked out`);
  }

  const action = actionInterface.legalActions[actionIndex];
  if (!action) {
    throw new Error(`No legal action at index ${actionIndex}`);
  }

  return action;
}

function legalActionIndices(actionInterface) {
  const mask = asArray(actionInterface && actionInterface.actionMask);
  const out = [];

  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1) {
      out.push(i);
    }
  }

  return out;
}

module.exports = {
  DEFAULT_MAX_ACTIONS,
  buildActionInterface,
  makeFixedActionMask,
  validateActionInterface,
  selectActionByIndex,
  legalActionIndices,
  sumMask,
};

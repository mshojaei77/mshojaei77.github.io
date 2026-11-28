const assert = require('node:assert/strict');
const { inspectText } = require('../assets/js/persian-text-inspector.js');
const { tokenize, rankDocuments } = require('../assets/js/bm25-playground.js');
const { estimateMemory } = require('../assets/js/llm-memory-estimator.js');
const { evaluateRetrieval } = require('../assets/js/retrieval-evaluation.js');
const { parseState, completionCount, schemaVersion } = require('../assets/js/learning-progress.js');

const inspected = inspectText('ك ي ١٢ 12\u200c\u200f', { letters: true, digits: true });
assert.equal(inspected.output, 'ک ی ۱۲ ۱۲\u200c\u200f');
assert.equal(inspected.changes.length, 2);
assert(inspected.characters.some(ch => ch.codePoint === 0x200c && ch.name.includes('NON-JOINER')));
assert(inspected.characters.some(ch => ch.codePoint === 0x200f && ch.name.includes('bidi control')));
assert.equal(inspectText('e\u0301', { nfc: true }).output, 'é');
assert.equal(inspectText('a\u200fب', { bidi: true }).output, 'aب');
assert.equal(inspectText('a\u200cب').output, 'a\u200cب');

assert.deepEqual(tokenize('ي ك۱۲ retrieval'), ['ی', 'ک۱۲', 'retrieval']);
const ranking = rankDocuments([{ id: 'a', text: 'cat cat sat' }, { id: 'b', text: 'cat dog' }], 'cat');
assert.equal(ranking.results[0].id, 'a');
assert.equal(ranking.results[0].terms[0].df, 2);
assert(Math.abs(ranking.results[0].score - 0.2373416716) < 1e-9);
assert.equal(rankDocuments([{ id: 'x', text: 'سلام فارسی' }], 'ناموجود').results[0].score, 0);
assert.throws(() => rankDocuments([], 'cat'), /at least one document/);
assert.throws(() => rankDocuments([{ text: 'x'.repeat(10001) }], 'x'), /under 10,000/);

const memory = estimateMemory({ parametersBillions: 1, weightBits: 8, batchSize: 1, contextLength: 10, layers: 2, kvHeads: 2, headDimension: 4, kvBytes: 2, runtimeOverheadPercent: 0, safetyMarginPercent: 0 });
assert.equal(memory.weights.bytes, 1e9);
assert.equal(memory.kvCache.bytes, 640);
assert.equal(memory.planningTotal.bytes, 1e9 + 640);
const groupedQuery = estimateMemory({ parametersBillions: 1, weightBits: 16, batchSize: 1, contextLength: 10, layers: 2, kvHeads: 1, headDimension: 4, kvBytes: 2, runtimeOverheadPercent: 10, safetyMarginPercent: 20 });
assert.equal(groupedQuery.kvCache.bytes, 320);
assert(Math.abs(groupedQuery.planningTotal.bytes - (2e9 + 320) * 1.1 * 1.2) < 1e-6);
assert.throws(() => estimateMemory({ parametersBillions: 1, weightBits: 7, batchSize: 1, contextLength: 10, layers: 2, kvHeads: 1, headDimension: 4, kvBytes: 2, runtimeOverheadPercent: 0, safetyMarginPercent: 0 }), /precision/);

const evaluated = evaluateRetrieval({ queries: [
  { id: 'q1', ranking: ['d2', 'd1', 'd3', 'd4'], relevance: { d1: 3, d2: 0, d3: 1, d4: 2 } },
  { id: 'q2', ranking: ['x1', 'x2'], relevance: { x1: 0, x2: 0 } }
] }, 2);
const q1 = evaluated.queries[0];
assert.equal(q1.precisionAtK, 0.5);
assert.equal(q1.recallAtK, 1 / 3);
assert.equal(q1.hitRateAtK, 1);
assert.equal(q1.reciprocalRank, 0.5);
assert(Math.abs(q1.dcgAtK - (7 / Math.log2(3))) < 1e-12);
const expectedNdcg = (7 / Math.log2(3)) / (7 + 3 / Math.log2(3));
assert(Math.abs(q1.ndcgAtK - expectedNdcg) < 1e-12);
assert(Math.abs(q1.averagePrecision - ((1 / 2) + (2 / 3) + (3 / 4)) / 3) < 1e-12);
assert.equal(evaluated.queries[1].reciprocalRank, 0);
assert.equal(evaluated.aggregate.mrr, 0.25);
const deduplicated = evaluateRetrieval({ queries: [{ ranking: ['a', 'a', 'b'], relevance: { a: 1 } }] }, 2);
assert.equal(deduplicated.queries[0].duplicates, 1);
assert.equal(deduplicated.queries[0].precisionAtK, 0.5);
assert.throws(() => evaluateRetrieval({ queries: [{ ranking: ['a'], relevance: { a: -1 } }] }), /grades/);
assert.throws(() => evaluateRetrieval({ queries: [{ ranking: ['a'], relevance: {} }] }, 0), /K must/);

const ids = ['one', 'two'];
const state = parseState(JSON.stringify({ schema_version: schemaVersion, completed: { one: true, stale: true } }), ids);
assert.equal(completionCount(state, ids), 1);
assert.deepEqual(parseState('{broken', ids).completed, {});
assert.deepEqual(parseState(JSON.stringify({ schema_version: 999, completed: { one: true } }), ids).completed, {});
console.log('Verified text inspection, BM25 scoring, memory formulas, retrieval metrics, and local progress schema.');

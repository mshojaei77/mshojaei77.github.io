const assert = require('node:assert/strict');
const { inspectText } = require('../assets/js/persian-text-inspector.js');
const { tokenize, rankDocuments } = require('../assets/js/bm25-playground.js');
const { estimateMemory } = require('../assets/js/llm-memory-estimator.js');
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

const ids = ['one', 'two'];
const state = parseState(JSON.stringify({ schema_version: schemaVersion, completed: { one: true, stale: true } }), ids);
assert.equal(completionCount(state, ids), 1);
assert.deepEqual(parseState('{broken', ids).completed, {});
assert.deepEqual(parseState(JSON.stringify({ schema_version: 999, completed: { one: true } }), ids).completed, {});
console.log('Verified text inspection, BM25 scoring, memory formulas, and local progress schema.');

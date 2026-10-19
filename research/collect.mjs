// Extract final memos and auditable tool activity from CommandCode NDJSON.
// Run from the project root: node research/collect.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';
import { createGunzip } from 'node:zlib';

const root = path.dirname(fileURLToPath(import.meta.url));
const runs = ['probe', 'foundations', 'systems', 'frontiers', 'audit', 'systems_recovery', 'synthesis', 'review_memos', 'frontiers_recovery'];
const manifest = [];
const activity = [];
for (const run of runs) {
  const file = path.join(root, 'raw', `${run}.jsonl`);
  if (!fs.existsSync(file) && !fs.existsSync(file + '.gz')) continue;
  const events = [];
  let currentText = '';
  const input = fs.existsSync(file) ? fs.createReadStream(file, { end: Math.max(0, fs.statSync(file).size - 1) }) : fs.createReadStream(file + '.gz').pipe(createGunzip());
  for await (const line of createInterface({ input, crlfDelay: Infinity })) {
    if (line.includes('"type":"message_start"')) currentText = '';
    if (line.includes('"type":"text_delta"')) {
      try { const item = JSON.parse(line); if (item.event?.type === 'text_delta') currentText += item.event.delta; } catch {}
      continue;
    }
    if (!/"type":"(?:result|tool_queued|tool_completed|tool_errored|model_request_end)"/.test(line)) continue;
    try { events.push(JSON.parse(line)); } catch { /* Incomplete live final line. */ }
  }
  const result = events.findLast(e => e.type === 'result');
  const requests = events.filter(e => e.event?.type === 'model_request_end');
  const calls = events.filter(e => e.event?.type === 'tool_queued');
  for (const e of events) {
    if (['tool_queued', 'tool_completed', 'tool_errored'].includes(e.event?.type)) {
      activity.push({ run, ...e.event });
    }
  }
  if (result?.finalText && run !== 'probe') {
    fs.writeFileSync(path.join(root, 'raw', `${run}.final.md`), result.finalText + '\n');
    if (!fs.existsSync(path.join(root, `${run}.md`))) fs.writeFileSync(path.join(root, `${run}.md`), result.finalText + '\n');
  } else if (currentText && run !== 'probe') {
    fs.writeFileSync(path.join(root, 'raw', `${run}.partial.md`), currentText);
  }
  manifest.push({
    run, status: result?.subtype ?? 'running', sessionId: result?.sessionId,
    models: [...new Set(requests.map(e => e.event.model))],
    turns: requests.length,
    currentResponseCharacters: currentText.length,
    webSearchCalls: calls.filter(e => e.event.toolName === 'web_search').length,
    webFetchCalls: calls.filter(e => e.event.toolName === 'web_fetch').length,
    toolErrors: events.filter(e => e.event?.type === 'tool_errored').length,
    usage: result?.usage, durationMs: result?.durationMs,
  });
}
fs.writeFileSync(path.join(root, 'run-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
fs.writeFileSync(path.join(root, 'tool-evidence.jsonl'), activity.map(e => JSON.stringify(e)).join('\n') + '\n');
console.log(JSON.stringify(manifest, null, 2));

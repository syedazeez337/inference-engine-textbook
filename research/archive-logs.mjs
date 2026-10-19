import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from 'node:stream/promises';
import { Transform, Writable } from 'node:stream';
import { createGzip, createGunzip } from 'node:zlib';
import { createHash } from 'node:crypto';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), 'raw');
const manifest = JSON.parse(fs.readFileSync(path.join(root, '..', 'run-manifest.json')));
for (const { run, status } of manifest) {
  if (!['success', 'error'].includes(status) || !/^[a-z_]+$/.test(run)) continue;
  const source = path.resolve(root, `${run}.jsonl`);
  if (path.dirname(source) !== root || !fs.existsSync(source)) continue;
  const archive = source + '.gz';
  const originalHash = createHash('sha256');
  const tap = new Transform({ transform(chunk, enc, cb) { originalHash.update(chunk); cb(null, chunk); } });
  await pipeline(fs.createReadStream(source), tap, createGzip({ level: 1 }), fs.createWriteStream(archive));
  const restoredHash = createHash('sha256');
  await pipeline(fs.createReadStream(archive), createGunzip(), new Writable({ write(chunk, enc, cb) { restoredHash.update(chunk); cb(); } }));
  if (originalHash.digest('hex') !== restoredHash.digest('hex')) throw new Error(`Archive verification failed: ${run}`);
  const before = fs.statSync(source).size;
  const after = fs.statSync(archive).size;
  fs.unlinkSync(source);
  console.log(`${run}: verified lossless archive, ${before} -> ${after} bytes`);
}

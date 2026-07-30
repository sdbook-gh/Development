#!/usr/bin/env node
const fontnik = require('fontnik');
const fs = require('fs');

const ttfPath = process.argv[2];
const start = parseInt(process.argv[3], 10);
const end = parseInt(process.argv[4], 10);
const outPath = process.argv[5];

const fontBuffer = fs.readFileSync(ttfPath);
fontnik.range({ font: fontBuffer, start, end }, (err, data) => {
  if (err) {
    process.stderr.write(err.message);
    process.exit(1);
  }
  fs.writeFileSync(outPath, data);
  process.stdout.write(`generated ${start}-${end} (${data.length} bytes)`);
});

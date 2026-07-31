import { readFileSync } from "node:fs";
import { parsePipelineTimes } from "./dependency-timeline.ts";
const path = process.argv[2];
if (!path) {
  console.error("usage: node dist/run.js <file>");
  process.exit(1);
}
const input = readFileSync(path, "utf8");
const result = parsePipelineTimes(input);
console.log(JSON.stringify(result, null, 2));

#!/bin/sh
# Run against a locally built image; no credentials or host mounts.
set -eu
: "${1:?usage: check-container-security.sh IMAGE}"
docker run --rm --pull=never --network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 64 --memory 512m --entrypoint node "$1" -e 'const a=require("node:assert/strict"),z=require("node:zlib");const [major,minor,patch]=process.versions.node.split(".").map(Number);a(major===22 && (minor>23 || minor===23 && patch>=4),process.version);const b=Buffer.from("Brotli regression");a.deepEqual(z.brotliDecompressSync(z.brotliCompressSync(b)),b);a.equal(z.constants.BROTLI_DECODER_ERROR_FORMAT_BLOCK_SWITCH,-17);console.log(process.version,"Brotli PASS")'

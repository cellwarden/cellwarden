#!/usr/bin/env bash
# Refuse a release tree that carries private material: personal paths and hostnames, internal
# notes, key-like strings or large data files. Run from the repository root:
#   tools/check_public_tree.sh            # checks the tracked files
set -uo pipefail
fail=0
say() { echo "check_public_tree: $*"; fail=1; }
files() { git ls-files; }

# private paths, hosts and internal names (the Iridis example page documents a site on purpose)
pat='/iridisfs/(ddnb|aihackathon)|/scratch/[a-z]{3,}[0-9]+|aazd1f17|loginX0[0-9]|ruby0[0-9]{2}|modal profile|progress\.md'
hits=$(git grep -n -I -E "$pat" -- . ':!tools/check_public_tree.sh' || true)
[ -n "$hits" ] && { say "private paths or hosts:"; echo "$hits" | head -20; }

# key-like strings (the test suite's fake key is allowed)
keys=$(git grep -n -I -E 'sk-ant-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9]{32,}|nvapi-[A-Za-z0-9_-]{20,}|dckr_pat_[A-Za-z0-9_-]{10,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}' -- . ':!tools/check_public_tree.sh' \
       | grep -v -E 'sk-ant-test-secret|nvapi-x+\b|nvapi-x{6,}' || true)
[ -n "$keys" ] && { say "key-like strings:"; echo "$keys"; }

# files that must not be published
for f in .env progress.md docs/PRESENTATION_HANDOFF.md; do
  files | grep -qx "$f" && say "internal file tracked: $f"
done
files | grep -E '(^|/)\.env$|\.pem$|id_rsa' && say "secret-like file tracked"

# large files (> 5 MB) other than what is allowed
while IFS= read -r f; do
  [ -f "$f" ] || continue
  s=$(stat -c %s "$f")
  [ "$s" -gt $((5 * 1024 * 1024)) ] && say "large file ($((s / 1048576)) MB): $f"
done < <(files)

[ $fail = 0 ] && echo "check_public_tree: ok ($(files | wc -l) files)"
exit $fail

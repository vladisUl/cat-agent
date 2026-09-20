#!/bin/bash

set -euo pipefail

cd /opt/cat-agent

branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
    echo "ERROR: detached HEAD"
    exit 1
fi

echo "== Local changes =="
git status --short

# Stage modifications and deletions of files already tracked by git.
# Untracked files are intentionally not added automatically.
git add -u

if ! git diff --cached --quiet; then
    message="${*:-Radxa local changes}"
    git commit -m "$message"
else
    echo "No tracked changes to commit."
fi

echo "== Sync with origin/$branch =="
git pull --rebase origin "$branch"
git push origin "$branch"

echo "== Done =="
git status --short --branch

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ "$PROJECT_DIR" == "/" || "$PROJECT_DIR" == "$HOME" ]]; then
    echo "ERROR: Refusing to run in root or home directory." >&2
    exit 1
fi

EXCLUDE_DIRS=(-path "*/.venv/*" -o -path "*/venv/*" -o -path "*/.env/*" -o -path "*/env/*" -o -path "*/.git/*")

safe_find_dirs() {
    find "$PROJECT_DIR" -type d -name "$1" \
        ! \( "${EXCLUDE_DIRS[@]}" \) \
        -print0
}

safe_find_files() {
    find "$PROJECT_DIR" -type f \( "$@" \) \
        ! \( "${EXCLUDE_DIRS[@]}" \) \
        -print0
}

count=0

remove_dirs() {
    local pattern="$1"
    while IFS= read -r -d '' dir; do
        echo "  rm -rf $dir"
        rm -rf "$dir"
        count=$((count + 1))
    done < <(safe_find_dirs "$pattern")
}

for pattern in "__pycache__" ".pytest_cache" ".mypy_cache" ".ruff_cache" \
               ".pytype" ".ipynb_checkpoints" ".tox" ".nox"; do
    remove_dirs "$pattern"
done

while IFS= read -r -d '' dir; do
    rm -rf "$dir"
    count=$((count + 1))
done < <(safe_find_dirs "*.egg-info")

while IFS= read -r -d '' file; do
    rm -f "$file"
    count=$((count + 1))
done < <(safe_find_files -name "*.pyc" -o -name "*.pyo" -o -name '*$py.class')

#!/usr/bin/env bash
# Show only episode start/end lines (with reward) from the HIL-SERL actor log.
# Usage:
#   scripts/watch_episodes.sh [path/to/actor_*.log]        # follow a log file
#   <actor command> 2>&1 | scripts/watch_episodes.sh -     # filter live output
# With no argument, follows the newest actor_*.log under outputs/.

pattern='\[EPISODE (START|END)\]|Episode reward'

src="${1:-}"
if [[ "$src" == "-" ]]; then
    exec grep --line-buffered -E "$pattern"
fi
if [[ -z "$src" ]]; then
    src=$(find outputs -name 'actor_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    [[ -z "$src" ]] && { echo "No actor_*.log found under outputs/; pass a path." >&2; exit 1; }
fi
echo "Following $src" >&2
tail -n +1 -F "$src" | grep --line-buffered -E "$pattern"

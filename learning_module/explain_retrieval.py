"""Explain what the retriever would inject, without injecting anything.

Reads the experience store accumulated by earlier phases, runs the real
retriever against a chosen state, and prints the ranking with every derived
quantity — decayed value, confidence, and the internal endowment-weighted score
that the injected text itself never carries.

Usage:
    python learning_module/explain_retrieval.py <db> [error_type ...]
    python learning_module/explain_retrieval.py <db> --top 1 test_failure
    python learning_module/explain_retrieval.py <db> --all
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning_module import classify, context, retrieve, value  # noqa: E402


def main(argv: list[str]) -> int:
    """Print the retrieval report for each requested state.

    @param argv - command-line arguments after the module name.
    @returns the process exit status.
    """
    try:
        # The report carries the rendered block verbatim, including its ✅/⚠️
        # markers, which a GBK console cannot encode.
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    args = list(argv)
    top = 5
    if '--top' in args:
        index = args.index('--top')
        top = int(args[index + 1])
        del args[index:index + 2]
    show_all = '--all' in args
    if show_all:
        args.remove('--all')
    if not args:
        print(__doc__)
        return 2

    database, *error_types = args
    if show_all or not error_types:
        error_types = list(classify.ERROR_KINDS)
    now = datetime.now(timezone.utc)

    print(f'store: {database}')
    print(f'now  : {now.isoformat()}')
    print(f'value params: alpha={value.ALPHA} lambda={value.LAMBDA}'
          f' half_life={value.HALF_LIFE_DAYS}d endowment={value.ENDOWMENT_WEIGHT}')
    print(f'floors: per-row confidence >= {value.MIN_CONFIDENCE},'
          f' best-of-top confidence >= {value.INJECTION_CONFIDENCE_FLOOR},'
          f' |effective| >= {value.MIN_EFFECTIVE_VALUE}')
    for error_type in error_types:
        result = retrieve.candidates_for(database, error_type, now=now)
        print(f'\n=== {error_type} ===')
        print(f'  searched kinds : {", ".join(result.error_types_searched)}')
        print(f'  exact matches  : {result.exact_matches}')
        print(f'  skipped stale  : {result.skipped_stale}')
        print(f'  skipped low conf: {result.skipped_low_confidence}')
        ranked = result.candidates
        if not ranked:
            print('  ranked         : (none)')
        else:
            print(f'  ranked (top {min(top, len(ranked))} of {len(ranked)}):')
            for candidate in ranked[:top]:
                print(f'    id={candidate.row_id:<3} {candidate.action_category:<14}'
                      f' ctx={candidate.tool_context:<12}'
                      f' n={candidate.total_count:<3}'
                      f' avg={candidate.average_reward:+.2f}'
                      f' age={candidate.age_days:6.1f}d'
                      f' decay={candidate.decay:.3f}'
                      f' effective={candidate.effective:+.3f}'
                      f' conf={candidate.confidence:.3f}'
                      f' adjusted={candidate.adjusted_score:+.3f}')
        # The injected text is the one Phase 3 renders: polarity-labelled and
        # budgeted, never the bare ranking above.
        rendered = context.render(result.entries)
        if rendered.text == '':
            print(f'  injection      : NONE — {result.reason or "no candidate survived the floors"}')
        else:
            print(f'  injection      : would inject {rendered.kept} entr(ies)'
                  f'{" (pre-cap applied)" if rendered.truncated else ""} —')
            for line in rendered.text.split('\n'):
                print(f'    {line}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))

"""Run every test module of the Small World test package.

Usage (from `app/`):

    python -m environments.smallw.tests.run_all
    python -m environments.smallw.tests.run_all test_classes test_engine

Every module named `test_*.py` in this package is imported and each of its
`test_*` functions is called; the exit status is 1 as soon as one fails.
The modules are also plain pytest files, so `pytest environments/smallw/tests`
works just as well.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback

PACKAGE = __package__ or 'environments.smallw.tests'


def discover() -> list[str]:
    """Names of the `test_*` modules of this package, sorted."""
    package = importlib.import_module(PACKAGE)
    return sorted(
        info.name for info in pkgutil.iter_modules(package.__path__)
        if info.name.startswith('test_')
    )


def run_module(name: str) -> tuple[int, list[str]]:
    """Run every `test_*` function of module `name`.

    Returns:
        (number of tests run, names of the failed ones).
    """
    module = importlib.import_module(f'{PACKAGE}.{name}')
    tests = [(attr, fn) for attr, fn in sorted(vars(module).items())
             if attr.startswith('test_') and callable(fn)]
    failed: list[str] = []
    print(f'\n== {name} ({len(tests)} tests) ==')
    for attr, fn in tests:
        try:
            fn()
        except Exception:                                   # noqa: BLE001
            failed.append(attr)
            print(f'FAIL {attr}')
            print(''.join('    ' + line for line in
                          traceback.format_exc().splitlines(keepends=True)))
        else:
            print(f'ok   {attr}')
    return len(tests), failed


def main(argv: list[str] | None = None) -> int:
    """Run the requested modules (all of them by default)."""
    names = list(argv) if argv else discover()
    if not names:
        print('no test module found')
        return 1
    total = 0
    failures: list[str] = []
    for name in names:
        ran, failed = run_module(name)
        total += ran
        failures.extend(f'{name}.{attr}' for attr in failed)
    print(f'\n{"=" * 60}')
    print(f'smallw tests: {total - len(failures)}/{total} passed '
          f'in {len(names)} module(s)')
    if failures:
        print('FAILED: ' + ', '.join(failures))
        return 1
    print('ALL GREEN')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

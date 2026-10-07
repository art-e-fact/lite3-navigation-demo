import pytest

try:
    from artefacts_toolkit.config import get_artefacts_params
except Exception:
    get_artefacts_params = None


def _parse_bool(value) -> bool:
    return str(value).strip().lower() in ('true', '1', 'yes', 'on')


def artefacts_params() -> dict:
    if get_artefacts_params is None:
        return {}
    try:
        return get_artefacts_params()
    except Exception:
        return {}


def pytest_addoption(parser):
    parser.addoption(
        '--headless', action='store_true', default=False,
        help='Run the simulator without a viewer',
    )


@pytest.fixture(scope='session')
def headless(request):
    if request.config.getoption('--headless'):
        return True
    return _parse_bool(artefacts_params().get('headless', 'false'))


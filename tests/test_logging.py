"""The package's public logging entry point.

``set_log_level`` is the one piece of setup the README asks users to call, and
its contract is mostly about what it does *not* do: it must not attach a second
handler when called again, and it must not disturb handlers the host
application configured. Both are invisible until they misbehave in someone
else's logging setup, so they are pinned here.
"""

import logging

import pytest

import s2mosaic

PACKAGE_LOGGER = "s2mosaic"


@pytest.fixture(autouse=True)
def restore_package_logger():
    """Leave the package logger exactly as the session found it."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    original_handlers = list(logger.handlers)
    original_level = logger.level
    yield
    logger.handlers = original_handlers
    logger.setLevel(original_level)


@pytest.fixture
def bare_logger():
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    return logger


def test_attaches_a_handler_and_sets_the_level(bare_logger):
    s2mosaic.set_log_level("INFO")

    assert len(bare_logger.handlers) == 1
    assert bare_logger.level == logging.INFO


def test_defaults_to_info(bare_logger):
    s2mosaic.set_log_level()

    assert bare_logger.level == logging.INFO


def test_accepts_a_level_constant(bare_logger):
    s2mosaic.set_log_level(logging.DEBUG)

    assert bare_logger.level == logging.DEBUG


def test_calling_again_updates_the_level_without_adding_a_handler(bare_logger):
    s2mosaic.set_log_level("INFO")
    handler = bare_logger.handlers[0]

    s2mosaic.set_log_level("DEBUG")

    assert bare_logger.handlers == [handler]
    assert bare_logger.level == logging.DEBUG


def test_leaves_a_host_application_handler_in_place(bare_logger):
    # An application that configured its own handler must keep it, and must
    # not have ours added on top of it, or every record it emits is doubled.
    host_handler = logging.NullHandler()
    bare_logger.addHandler(host_handler)

    s2mosaic.set_log_level("WARNING")

    assert bare_logger.handlers == [host_handler]
    assert bare_logger.level == logging.WARNING


def test_emits_records_at_the_configured_level(bare_logger, capsys):
    s2mosaic.set_log_level("INFO")

    logging.getLogger("s2mosaic.testing").info("hello from the pipeline")
    logging.getLogger("s2mosaic.testing").debug("not at this level")

    stderr = capsys.readouterr().err
    assert "hello from the pipeline" in stderr
    assert "not at this level" not in stderr


def test_progress_logs_are_silent_before_it_is_called(bare_logger, capsys):
    # Standard library convention, and what the README leans on: the pipeline
    # stage logs stay quiet until the host asks for them. Scoped to INFO
    # deliberately. Warnings and above still reach stderr through logging's
    # ``lastResort`` handler when nothing is configured, which is the
    # behaviour we want and not something this call controls.
    logging.getLogger("s2mosaic.testing").info("progress should not appear")

    assert capsys.readouterr().err == ""

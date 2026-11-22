"""Create sub-tab empty state (user report 2026-11-22).

Opening the Create tab before generating anything on Imagine showed a blank
page: ``result_group`` is hidden until generation starts and nothing else
lived in the tab. The tab now carries a tooltip and an in-tab hint that is
visible exactly while ``result_group`` is hidden -- including when another
caller (the dialog's session restore) toggles ``result_group`` directly and
when the Create tab is not the current tab.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from myvoice.ui.dialogs.voice_design_studio.description_path_panel import (  # noqa: E402
    DescriptionPathPanel,
)

CREATE = 1


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qapp):
    p = DescriptionPathPanel()
    p.show()
    yield p
    p.hide()
    p.deleteLater()


def test_create_tab_has_tooltip_pointing_to_imagine(panel):
    tip = panel.sub_tabs.tabToolTip(CREATE)
    assert "Imagine" in tip and "Generate 5 Variations" in tip


def test_hint_shown_before_any_generation(panel):
    panel.sub_tabs.setCurrentIndex(CREATE)
    assert panel.result_group.isHidden()
    assert panel.create_empty_label.isVisible()


def test_hint_hides_when_results_shown_and_returns_on_clear(panel):
    panel.sub_tabs.setCurrentIndex(CREATE)
    panel.result_group.setVisible(True)  # as the generation-start path does
    assert not panel.create_empty_label.isVisible()

    panel._clear_result()
    assert panel.create_empty_label.isVisible()


def test_hint_tracks_results_toggled_while_create_tab_not_current(panel):
    """Session restore sets result_group visible from the dialog, possibly
    while Imagine is the current tab (no Show event reaches the group)."""
    panel.sub_tabs.setCurrentIndex(0)
    panel.result_group.setVisible(True)
    panel.sub_tabs.setCurrentIndex(CREATE)
    assert not panel.create_empty_label.isVisible()
    assert panel.result_group.isVisible()

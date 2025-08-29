import pytest
from qtpy.QtCore import Qt

from ilastikrag.gui.feature_selection_dialog import FeatureSelectionDialog


@pytest.fixture
def channel_names():
    return ["Grayscale", "Membranes"]


@pytest.fixture
def feature_names():
    return [
        "standard_edge_mean",
        "standard_edge_maximum",
        "standard_edge_count",
        "standard_sp_mean",
        "standard_sp_maximum",
        "standard_sp_count",
        "standard_edge_quantiles_10",
        "standard_edge_quantiles_90",
    ]


@pytest.fixture
def initial_selections():
    return {
        "Grayscale": ["standard_edge_mean"],
        "Membranes": [
            "standard_edge_quantiles_10",
            "standard_edge_quantiles_90",
        ],
    }


@pytest.fixture
def default_selections():
    return {
        "Grayscale": ["standard_edge_maximum"],
        "Membranes": [
            "standard_edge_count",
            "standard_edge_quantiles_90",
            "standard_sp_mean",
        ],
    }


def test_no_features(qtbot, channel_names, feature_names):
    dlg = FeatureSelectionDialog(channel_names, feature_names)
    qtbot.addWidget(dlg)
    dlg.accept()

    assert dlg.selections() == {"Grayscale": [], "Membranes": []}


def test_initial_selection(qtbot, channel_names, feature_names, initial_selections):
    dlg = FeatureSelectionDialog(channel_names, feature_names, initial_selections)
    qtbot.addWidget(dlg)
    dlg.accept()

    assert dlg.selections() == initial_selections


def test_reset_to_defaults(
    qtbot, channel_names, feature_names, initial_selections, default_selections
):
    dlg = FeatureSelectionDialog(
        channel_names, feature_names, initial_selections, default_selections
    )
    qtbot.addWidget(dlg)
    qtbot.mouseClick(dlg._resetButton, Qt.LeftButton)
    dlg.accept()

    assert dlg.selections() != initial_selections
    assert dlg.selections() == default_selections

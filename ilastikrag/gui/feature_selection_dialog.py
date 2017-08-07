from collections import OrderedDict
from itertools import groupby

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QDialogButtonBox

from util import HierarchicalChecklistView, Checklist

# FIXME: These imports will be needed if we ever figure out how to manage the sizes properly...
#from PyQt5.QtCore import QSize
#from PyQt5.QtWidgets import QSizePolicy, QScrollArea

class FeatureSelectionDialog(QDialog):
    """
    A UI for selecting edge features on a per-channel basis.
    
    Example usage:
    
    .. code-block:: python
    
        channel_names = ['Grayscale', 'Membranes']
        feature_names = ['standard_edge_mean', 'standard_edge_count',
                         'standard_sp_mean', 'standard_sp_count',
                         'standard_edge_quantiles', 'standard_edge_quantiles_10', 'standard_edge_quantiles_90']

        default_selections = { 'Grayscale': ['standard_sp_mean', 'standard_sp_count'],
                               'Membranes': ['standard_edge_quantiles'] }

        dlg = FeatureSelectionDialog(channel_names, feature_names, default_selections)
        dlg.exec_()
        if dlg.exec_() == QDialog.Accepted:
            print dlg.selections()
    """
    def __init__(self, channel_names, feature_names, default_selections=None, parent=None):
        """
        Parameters
        ----------
        channel_names
            *list of str*
            The user will be shown a separate checklist of feature options for each channel.
        
        feature_names
            *list of str*
            Feature names, exactly as expected by :py:meth:`~ilastikrag.rag.Rag.compute_features()`.
            The features will be grouped by category and shown in duplicate checklist widgets for each channel.
        
        default_selections
            *dict, str: list-of-str*
            Mapping from channel_name -> feature_names, indicating which
            features should be selected by default for each channel.
        
        parent
            *QWidget*
        """
        super(FeatureSelectionDialog, self).__init__(parent)
        
        self.setWindowTitle("Select Edge Features")
        self.tree_widgets = {}

        self.checklist_widgets = OrderedDict()
        boxes_layout = QHBoxLayout()
        for channel_name in channel_names:
            default_checked = []
            if default_selections and channel_name in default_selections:
                default_checked = default_selections[channel_name]
            checklist = _make_checklist(feature_names, default_checked)
            checklist.name = channel_name
            checklist_widget = HierarchicalChecklistView( checklist, parent=self )
            self.checklist_widgets[channel_name] = checklist_widget
            boxes_layout.addWidget(checklist_widget)

        buttonbox = QDialogButtonBox( Qt.Horizontal, parent=self )
        buttonbox.setStandardButtons( QDialogButtonBox.Ok | QDialogButtonBox.Cancel )
        buttonbox.accepted.connect( self.accept )
        buttonbox.rejected.connect( self.reject )

        widget_layout = QVBoxLayout()

        # FIXME: Would like to hold the TreeWidgets in a QScrollArea,
        #        but they don't seem to respect fixed size policies,
        #        so the scroll area never shows a scrollbar...
        #scrollarea = QScrollArea()
        #scrollarea.setLayout(boxes_layout)
        #widget_layout.addWidget(scrollarea)
        widget_layout.addLayout(boxes_layout)

        widget_layout.addWidget(buttonbox)
        self.setLayout(widget_layout)

        total_spacing = self.width() - (len(channel_names)*checklist_widget.width())
        total_width = total_spacing + len(channel_names) * ( 20 + checklist_widget.columnWidth(0) )
        self.resize(total_width, 500)

    def selections(self):
        """
        Return the user's choices as a dictionary (channel -> feature names).
        Call this after the dialog has been accepted.
        """
        selections = {}
        for channel_name, checklist_widget in self.checklist_widgets.items():
            channel_selections = []
            nested_selections = checklist_widget.checklist.to_nested_dict(filter_by_checkstate=True)
            for _category_name, category_dict in nested_selections.items():
                for _subgroup_name, subgroup_items in category_dict.items():
                    for _short_name, (checkstate, feature_name) in subgroup_items.items():
                        assert checkstate is True
                        channel_selections.append( feature_name )
            selections[channel_name] = channel_selections
        return selections

    @classmethod
    def launch(cls, channel_names, feature_names, default_selections=None):
        from PyQt5.QtWidgets import QApplication
        if QApplication.instance() is None:
            app = QApplication([])
        
        dlg = FeatureSelectionDialog(channel_names, feature_names, default_selections)
        dlg.show()
        dlg.raise_()
        dlg.exec_()
        if dlg.result() == QDialog.Accepted:
            return dlg.selections()
        return None

def _make_checklist(feature_names, default_checked):
    feature_groups = _group_features(feature_names)

    cat_checklists = []
    for category, category_group in feature_groups.items():
        subgroup_checklists = []
        for subgroup_name, subgroup in category_group.items():
            feature_checklist_items = []
            for feature_name in subgroup:
                # Instead of looking for an exact match, 
                # we use 'any' here to auto-check all quantiles if the user
                # just gave e.g. 'standard_sp_quantiles'
                checkstate = any(feature_name.startswith(checked) for checked in default_checked)
                feature_checklist_items.append( Checklist(feature_name.split('_')[-1], checkstate, None, feature_name) )
            subgroup_checklists.append( Checklist(subgroup_name, Qt.Unchecked, feature_checklist_items, None ) )
        cat_checklists.append( Checklist( category, Qt.Unchecked, subgroup_checklists, None ) )

    return Checklist( 'root', Qt.Unchecked, cat_checklists, None )
    
def _group_features(feature_names):
    """
    Build up a big dict-of-dicts indexed like this:
    feature_groups[(category, type)][subgroup] = [subgroup_features...]
    
    For example:
    
        feature_groups['standard (sp)']['general']     = ['standard_sp_count', 'standard_sp_mean', ...]
        feature_groups['standard (sp)']['regionradii'] = ['standard_sp_regionradii_0', 'standard_sp_regionradii_1']
        feature_groups['standard (sp)']['regionaxes']  = ['standard_sp_regionaxes_0x', 'standard_sp_regionaxes_0y',
                                                          'standard_sp_regionaxes_1x', 'standard_sp_regionaxes_1y']
    """

    feature_groups = OrderedDict()
    feature_name_tuples = tuple(name.split('_') for name in sorted(feature_names) )
    for (feature_category, feature_type), group in groupby(feature_name_tuples, lambda tup: tup[:2] ):
        subgroups = OrderedDict()
        subgroups['general'] = []
        for subgroup_name, subgroup in groupby(group, lambda tup: tup[2]):
            # We assume the name 'general' isn't part of any feature names,
            # so we can use it as a catch-all subgroup for this widget
            assert subgroup_name != 'general'

            subgroup_feature_names = map( '_'.join, subgroup )
            if len(subgroup_feature_names) == 1:
                subgroups['general'] += subgroup_feature_names
            else:
                # Drop the 'top-level' feature name, e.g. 'standard_edge_quantiles',
                # keeping only the 'low-level' names like 'standard_edge_quantiles_10'
                subgroup_feature_names = filter( lambda name: len(name.split('_')) == 4, subgroup_feature_names )
                subgroups[subgroup_name] = subgroup_feature_names
        
        if len(subgroups['general']) == 0:
            del subgroups['general']
        feature_groups["{} ({})".format(feature_category, feature_type)] = subgroups
    return feature_groups

if __name__ == "__main__":
    # Make the program quit on Ctrl+C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    #channel_names = ['Grayscale', 'Membranes', 'Cytoplasm', 'Mitochondria']
    channel_names = ['Grayscale', 'Membranes']
    feature_names = ['standard_edge_mean', 'standard_edge_maximum', 'standard_edge_count',
                     'standard_sp_mean', 'standard_sp_maximum', 'standard_sp_count',
                     'standard_edge_quantiles', 'standard_edge_quantiles_10', 'standard_edge_quantiles_90']

    default_selections = { 'Grayscale': ['standard_sp_mean', 'standard_sp_count'],
                           'Membranes': ['standard_edge_quantiles'] }

    selections = FeatureSelectionDialog.launch(channel_names, feature_names, default_selections)
    print selections

#     from PyQt5.QtWidgets import QApplication
#     app = QApplication([])
# 
# 
#     dlg = FeatureSelectionDialog(
#     dlg.show()
#     dlg.raise_()
#     dlg.exec_()
# 
#     print dlg.selections()

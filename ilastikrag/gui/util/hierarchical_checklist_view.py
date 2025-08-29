from collections import OrderedDict
from functools import partial
from contextlib import contextmanager

from qtpy.QtCore import Qt, QEvent
from qtpy.QtGui import QStandardItemModel, QStandardItem
from qtpy.QtWidgets import QTreeView, QStyledItemDelegate, QCheckBox


class HierarchicalChecklistView(QTreeView):
    """
    Given a hierarchical 'checklist' data structure (see below),
    loads the checklist into a TreeView, in which each item is a QCheckBox.
    Checking a parent item auto-checks (or unchecks) all of its children.
    
    (This would not be straightforward to implement using a simple QTreeWidget,
    so this class is specially designed just for this task.)
    """
    def __init__(self, checklist, parent=None):
        super(HierarchicalChecklistView, self).__init__(parent)
        self.setModel(checklist)

    def open_persistent_editors_for_children(self, item):
        model_index = self.model().indexFromItem(item)
        for row in range(self.model().rowCount(model_index)):
            child_item = item.child(row, 0)
            self.openPersistentEditor(self.model().indexFromItem(child_item))
            self.open_persistent_editors_for_children(child_item)

    @property
    def checklist(self):
        return self.model()._checklist
    
    def keyPressEvent(self, event):
        # We ignore events for all keys,
        # so that our parent widget can handle those.
        # (E.g. QDialog can accept/reject the dialog when the user presses 'enter' or 'esc'.)
        if  event.type() == QEvent.KeyPress:
            event.ignore()

    def setModel(self, checklist):
        super().setModel(HierarchicalChecklistModel(checklist))
        self.setItemDelegate(HierarchicalChecklistViewDelegate(self))

        # Open a persistent editor for every item
        self.open_persistent_editors_for_children(self.model().invisibleRootItem())
        self.expandAll()
        self.model().setHorizontalHeaderLabels([self.model()._checklist.name])

class Checklist(object):
    """
    The view's real 'model data' is stored in this Checklist class,
    not in QStandardItem.data()
    """
    def __init__(self, name, checkstate, children=None, data=None):
        children = children or None # Use None, not []
        if checkstate in (True, Qt.Checked):
            checkstate = Qt.Checked
        else:
            checkstate = Qt.Unchecked
        
        self.name = name
        self.checkstate = checkstate
        self.children = children
        self.data = data or name
        self.autoset_checkstate_from_children()

    def autoset_checkstate_from_children(self):        
        if not self.children:
            return

        all_children_checked = all(child.checkstate == Qt.Checked for child in self.children)
        all_children_unchecked = all(child.checkstate == Qt.Unchecked for child in self.children)

        if all_children_checked:
            self.checkstate = Qt.Checked
        elif all_children_unchecked:
            self.checkstate = Qt.Unchecked
        else:
            self.checkstate = Qt.PartiallyChecked

    def get_descendent(self, checklist_indexes):
        if not checklist_indexes:
            return self
        child = self.children[checklist_indexes[0]]
        return child.get_descendent(checklist_indexes[1:])
    
    @classmethod
    def from_tuples(cls, tuples):
        """
        """
        if len(tuples) == 4:
            name, checkstate, children, data = tuples
        else:
            name, checkstate, children = tuples
            data = name
        assert isinstance(name, str)
        assert not children or isinstance(children, list)
        
        if children:
            children = list(map(cls.from_tuples, children))
        return Checklist(name, checkstate, children, data)

    def to_tuples(self, filter_by_checkstate=None):
        return_as_bool = False
        if filter_by_checkstate is True:
            filter_by_checkstate = Qt.Checked
            return_as_bool = True
        if filter_by_checkstate is False:
            filter_by_checkstate = Qt.Unchecked
            return_as_bool = True

        if not self.children:
            return (self.name, self.checkstate, None, self.data)

        child_tuples = [c.to_tuples(filter_by_checkstate) for c in self.children]

        if filter_by_checkstate is not None:
            filtered_child_tuples = []
            for child_tuple in child_tuples:
                has_children = child_tuple[2]
                if has_children or child_tuple[1] == filter_by_checkstate:
                    filtered_child_tuples.append( child_tuple )
            child_tuples = filtered_child_tuples

        def convert_checkstates(tuples):
            child_tuples = tuples[2]
            if child_tuples:
                child_tuples = list(map(convert_checkstates, child_tuples))
            
            checkstate = (tuples[1] != Qt.Unchecked)
            return (tuples[0], checkstate, child_tuples, tuples[3])

        result = (self.name, self.checkstate, child_tuples, self.data)
        if return_as_bool:
            return convert_checkstates(result)
        return result
    
    def to_nested_dict(self, filter_by_checkstate=None):
        """
        Convert this Checklist into a nested dictionary.
        (The 'root' level is omitted.)
        
            For example:
                tuples = ('root', False, [
                             ('first', True, None),
                             ('second', False, None),
                             ('third', False, [
                                 ('third-first', True, None),
                                 ('third-second', False, None)
                             ]),
                             ('fourth', False, None),
                          ])
            
                checklist = Checklist.from_tuples() 
            
                d = checklist.to_nested_dict()
                assert d['first'][0] == Qt.Checked
                assert d['second'][0] == Qt.Unchecked
                assert d['third']['third-first'][0] == Qt.Checked
                assert d['third']['third-second'][0] == Qt.Unchecked
                assert d['fourth'][0] == Qt.Unchecked

        If filter_by_checkstate is given, leaf nodes that don't match the given checkstate will be returned.
        Acceptable values for filter_by_checkstate include: Qt.Checked/Unchecked, True/False.
        If True or False is given, the returned tuples will be converted to bool.
        """
        return_as_bool = False
        if filter_by_checkstate is True:
            filter_by_checkstate = Qt.Checked
            return_as_bool = True
        if filter_by_checkstate is False:
            filter_by_checkstate = Qt.Unchecked
            return_as_bool = True

        if not self.children:
            return (self.checkstate, self.data)

        d = OrderedDict()
        for child in self.children:
            if (filter_by_checkstate is None) or (child.children or child.checkstate == filter_by_checkstate):
                d[child.name] = child.to_nested_dict(filter_by_checkstate)                

        for name in list(d.keys()):
            if isinstance(d[name], dict) and len(d[name]) == 0:
                del d[name]

        def convert_checkstates(dd):
            for key in dd.keys():
                if isinstance(dd[key], dict):
                    convert_checkstates(dd[key])
                else:
                    checkstate, data = dd[key]
                    dd[key] = ((checkstate != Qt.Unchecked), data)

        if return_as_bool:
            convert_checkstates(d)
        return d

class HierarchicalChecklistModel(QStandardItemModel):
    
    def __init__(self, checklist):
        super(HierarchicalChecklistModel, self).__init__()
        assert isinstance(checklist, Checklist)
        self._checklist = checklist
        self._append_to_standard_item(self.invisibleRootItem(), checklist.children)
        self._updating_paused = False

    def data(self, index, role=Qt.DisplayRole):
        # We don't want to let the subclass display the data, because we'll
        # just create a bunch of 'editors' (checkboxes) which will show the data.
        if role == Qt.DisplayRole:
            return ''
        return super(HierarchicalChecklistModel, self).data(index, role)

    def _append_to_standard_item(self, item, checklist):
        if checklist is not None:
            for child_checklist in checklist:
                childitem = QStandardItem(child_checklist.name)
                childitem.setFlags( Qt.ItemIsEnabled )
                item.appendRow([childitem])
                self._append_to_standard_item(childitem, child_checklist.children)

    def get_checklist_from_model_index(self, model_index):
        nested_checklist_indexes = ()
        while model_index.isValid():
            nested_checklist_indexes = (model_index.row(),) + nested_checklist_indexes
            model_index = model_index.parent()
        return self._checklist.get_descendent(nested_checklist_indexes)

    @contextmanager
    def pause_updates(self):
        self._updating_paused = True
        yield
        self._updating_paused = False

    def handle_checkstate(self, model_index, checkstate):
        if self._updating_paused:
            return # This change wasn't triggered by the user

        if checkstate == Qt.PartiallyChecked:
            checkstate = Qt.Checked
            checklist = self.get_checklist_from_model_index(model_index)
            checklist.checkstate = checkstate
            self.dataChanged.emit(model_index, model_index)

        self._update_child( model_index, checkstate )
        self._update_parent( model_index.parent() )

    def _update_child(self, model_index, checkstate):
        checklist = self.get_checklist_from_model_index(model_index)
        checklist.checkstate = checkstate
        if checklist.children:
            for row in range(len(checklist.children)):
                child_index = model_index.child(row, 0)
                self._update_child(child_index, checkstate)
        self.dataChanged.emit(model_index, model_index)
    
    def _update_parent(self, model_index):
        if not model_index.isValid():
            return        
        checklist = self.get_checklist_from_model_index(model_index)
        checklist.autoset_checkstate_from_children()
        self._update_parent( model_index.parent() )
        self.dataChanged.emit(model_index, model_index)

class HierarchicalChecklistViewDelegate(QStyledItemDelegate):
    
    def createEditor(self, parent, option, model_index):
        assert isinstance(self.parent(), HierarchicalChecklistView)
        checklist = self.parent().model().get_checklist_from_model_index(model_index)
        checkbox = QCheckBox(checklist.name, parent=parent)
        checkbox.setTristate(True)
        checkbox.stateChanged.connect(partial(self.parent().model().handle_checkstate, model_index))
        return checkbox

    def setEditorData(self, editor, model_index):
        assert isinstance(editor, QCheckBox)
        checkbox = editor
        checklist = self.parent().model().get_checklist_from_model_index(model_index)
        checkbox.setText(checklist.name)
        
        # Pause updates: We don't want to model to react to
        # programmatic changes, only actual user clicks.
        with self.parent().model().pause_updates():
            checkbox.setCheckState(checklist.checkstate)

if __name__ == "__main__":
    # Make the program quit on Ctrl+C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    from qtpy.QtWidgets import QApplication
    app = QApplication([])
    
    example = ('root', False, [
                 ('first', True, None),
                 ('second', False, None),
                 ('third', False, [
                     ('third-first', True, None),
                     ('third-second', False, None)
                 ]),
                 ('fourth', False, None),
              ])

    checklist = Checklist.from_tuples(example)

    treeview = HierarchicalChecklistView(checklist)    
    treeview.show()
    treeview.raise_()
    
    app.exec_()

    print(treeview.checklist)
    print(treeview.checklist.to_tuples())
    print(treeview.checklist.to_nested_dict())

    d = treeview.checklist.to_nested_dict(filter_by_checkstate=True)
    print(d)
    print("")
    print("")

    d = treeview.checklist.to_nested_dict()
    assert d['first'][0] == Qt.Checked
    assert d['second'][0] == Qt.Unchecked
    assert d['third']['third-first'][0] == Qt.Checked
    assert d['third']['third-second'][0] == Qt.Unchecked
    assert d['fourth'][0] == Qt.Unchecked

    d = treeview.checklist.to_nested_dict(filter_by_checkstate=True)
    print(d)
    assert d['first'][0] == True
    assert 'second' not in d
    assert d['third']['third-first'][0] == True
    assert 'third-second' not in d['third']
    assert 'fourth' not in d

    t = treeview.checklist.to_tuples(filter_by_checkstate=True)
    assert t[2][0][0] == 'first'
    assert t[2][0][1] == True

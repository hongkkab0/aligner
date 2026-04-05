#!/usr/bin/env python
# -*- coding: utf8 -*-
from __future__ import annotations

import codecs
import os.path
import subprocess
import time
from functools import partial
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import json
from typing import List, Set, Dict, Tuple

import aligner_gui.labeler.resources
# Add internal libs
from aligner_gui.labeler.libs.label_manager import LabelManager
from aligner_gui.labeler.libs.lib import struct, newAction, newIcon, addActions, fmtShortcut, KeptBoxAction
from aligner_gui.labeler.libs.shape import Shape
from aligner_gui.labeler.libs.canvas import Canvas
from aligner_gui.labeler.libs.zoomWidget import ZoomWidget
from aligner_gui.labeler.libs.toolBar import ToolBar
from aligner_gui.labeler.libs.ustr import ustr
from aligner_gui import __appname__
from aligner_gui.labeler import utils as util
from aligner_gui.project.project_dataset_service import (
    build_dataset_summary,
    inspect_image_labels,
    load_labeler_image_list,
    save_labeler_image_list,
)
from aligner_gui.labeler.file_list_service import remove_paths_from_file_list
from aligner_gui.utils.image_cache import CachedImageReader, decode_image_with_cv2
import logging
import math
from copy import deepcopy

# auto labeling
import aligner_engine.const as const
from shutil import copyfile
import traceback
from aligner_gui.utils import gui_util


class HashableQListWidgetItem(QListWidgetItem):

    def __init__(self, *args):
        super(HashableQListWidgetItem, self).__init__(*args)

    def __hash__(self):
        return hash(id(self))


class ImageIndexThread(QThread):
    sig_progress = pyqtSignal(int, int, str)
    sig_completed = pyqtSignal(object, bool)
    sig_failed = pyqtSignal(str)

    def __init__(self, img_paths):
        super().__init__()
        self._img_paths = list(img_paths)
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            states = inspect_image_labels(
                self._img_paths,
                progress_callback=lambda cur, total, path: self.sig_progress.emit(cur, total, path),
                should_stop=lambda: self._cancel_requested,
            )
            self.sig_completed.emit(states, self._cancel_requested)
        except Exception as exc:
            self.sig_failed.emit(str(exc))


class LabelerWidget(QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))
    IMAGE_LIST_BATCH_SIZE = 250
    ZOOM_WHEEL_FACTOR = 1.05

    def __init__(self, main_window, session, project_path, is_new: bool):
        super(LabelerWidget, self).__init__()
        self.setWindowTitle(__appname__)
        # For loading all image under a directory
        from aligner_gui.main_window import MainWindow
        self._main_window: MainWindow = main_window
        self._image_paths: List[str] = []
        self._dir_name = None
        self._project_path: str = project_path
        self._worker = session
        self._image_index_thread: ImageIndexThread | None = None
        self._image_index_progress: QProgressDialog | None = None
        self._pending_image_states = []
        self._pending_image_state_index = 0
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(250)
        self._autosave_timer.timeout.connect(self._flush_autosave)
        self._image_reader = CachedImageReader(self._decode_image_mat, max_items=8)

        self.lastOpenDir = None
        self._is_new = is_new
        self._copied_shape: Shape = None

        # Whether we need to save or not.
        self._dirty = False

        self.isEnableCreate = True
        self.isEnableCreateRo = True

        # Enble auto saving if pressing next
        self._autoSaving = True
        self._noSelectionSlot = False
        self.screencastViewer = "firefox"
        self.screencast = "https://youtu.be/7D5lvol_QRA"
        # For a demo of original labelImg, please see "https://youtu.be/p0nR2YsCY_U"
        LabelManager.init()

        self._label_list_items_to_shapes: [Dict[HashableQListWidgetItem, Shape]] = {}
        self._shapes_to_label_list_items: [Dict[Shape, HashableQListWidgetItem]] = {}
        self._prevLabelText = ''

        ## right upper dock
        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self._label_list_widget = QListWidget()
        self._label_list_widget.setUniformItemSizes(True)
        self._label_list_widget.itemActivated.connect(self._label_list_selection_changed)
        self._label_list_widget.itemSelectionChanged.connect(self._label_list_selection_changed)
        self._label_list_widget.itemDoubleClicked.connect(self._edit_label_name)
        self._label_list_widget.itemChanged.connect(self._label_list_name_edited)

        self._lbl_label_info = QLabel(self)
        self._lbl_label_info.setLineWidth(-1)
        self._lbl_label_info.setText("")
        self._lbl_label_info.setObjectName("lbl_label_info")

        listLayout.addWidget(self._label_list_widget)
        listLayout.addWidget(self._lbl_label_info)

        self.dock = QDockWidget(u'Label List', self)
        self.dock.setObjectName(u'Label')
        self.dock.setWidget(labelListContainer)

        self._file_list_widget = QListWidget()
        self._file_list_widget.setUniformItemSizes(True)
        self._file_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._file_list_widget.setItemDelegate(ElideLeftDelegate())
        self._file_list_widget.itemDoubleClicked.connect(self._file_list_item_clicked)
        self._file_list_widget.itemClicked.connect(self._file_list_item_clicked)
        self._file_list_widget.itemSelectionChanged.connect(self._item_selected_changed_file_list)

        self._file_list_total_count = 0
        self._file_list_labeled_count = 0
        self._labeled_info_dict: Dict[str, bool] = {}  # key: img_path, val: True, if It is labeled.
        self._lbl_file_list_info = QLabel(self)
        self._lbl_file_list_info.setLineWidth(-1)
        self._lbl_file_list_info.setText("")
        self._lbl_file_list_info.setObjectName("lbl_file_list_info")



        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self._file_list_widget)
        filelistLayout.addWidget(self._lbl_file_list_info)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(u'File List', self)
        self.filedock.setObjectName(u'File')
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()

        self.canvas = Canvas(self)
        self.canvas.sig_zoom_request.connect(self.zoomRequest)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.canvas.sig_scroll_request.connect(self.scrollRequest)
        self.canvas.moveViewPointHorizontalRequest.connect(self.moveViewPointHorizontalRequest)
        self.canvas.moveViewPointVerticalRequest.connect(self.moveViewPointVerticalRequest)
        self.canvas.moveViewPointPressed.connect(self.moveViewPointPressed)

        self.canvas.sig_new_shape_made.connect(self._new_shape_made_from_canvas)
        self.canvas.shapeMoved.connect(self._set_dirty)
        self.canvas.sig_shape_selection_changed.connect(self._canvas_shape_selection_changed)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.status.connect(self.status)

        self.canvas.hideNRect.connect(self.enableCreate)
        self.canvas.hideRRect.connect(self.enableCreateRo)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        # Tzutalin 20160906 : Add file list and dock to move faster
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.dockFeatures = QDockWidget.DockWidgetClosable \
                            | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)
        self.filedock.setFeatures(self.filedock.features() ^ self.dockFeatures)

        # Actions
        action = partial(newAction, self)

        quit = action('&Quit', self.close,
                      'Ctrl+Q', 'quit', u'Quit application')

        open = action('&Open', self.openFile,
                      'Ctrl+O', 'open', u'Open image or label file')

        opendir = action('&Open Dir', self._open_dir,
                         'Ctrl+u', 'open', u'Open Dir')

        openNextImg = action('&Next Image', self.openNextImg,
                             '', 'next', u'Open Next')

        openPrevImg = action('&Prev Image', self.openPrevImg,
                             '', 'prev', u'Open Prev')

        save = action('&Save', self._save_label,
                      'Ctrl+S', 'save', u'Save labels to file')
        save_selected = action(
            'Save\nSelected',
            self._save_selected_labels,
            'Ctrl+Shift+S',
            'save',
            u'Save current labels to all selected images',
        )
        close = action('&Close', self.closeFile,
                       'Ctrl+W', 'close', u'Close current file')

        createMode = action('Create\nRotatedBox', self.setCreateMode,
                            'F1', 'new', u'Start drawing Boxs', enabled=False)
        editMode = action('&Edit\nRotatedBox', self.setEditMode,
                          'F4', 'edit', u'Move and edit Boxs', enabled=False)

        # create = action('Create\nRectBox', self.createShape,
        #                 'w', 'new', u'Draw a new Box', enabled=False)
        #
        # createRo = action('Create\nRotatedRBox', self.createRoShape,
        #                 'e', 'newRo', u'Draw a new RotatedRBox', enabled=False)

        keep = action('&Keep\nBox', self._keep_longside_shortside_of_box,
                      'Ctrl+Shift+C', 'copy', u'Keep this box into kept box', enabled=False)
        apply = action('&Apply\nKept\nBox', self._apply_longside_shortside_of_kept_box,
                      'Ctrl+Shift+V', 'copy', u'Apply kept box to this box', enabled=False)
        delete = action('Delete\nBox', self._delete_selected_shape,
                        'Delete', 'delete', u'Delete', enabled=False)
        copy = action('&Copy\nBox', self._copy_selected_shape,
                      'Ctrl+C', 'copy', u'Copy the selected Box', enabled=False)
        paste = action('&Paste\nBox', self._paste_copied_shape,
                       'Ctrl+V', 'copy', u'Paste the copied Box')

        # advancedMode = action('&Advanced Mode', self.toggleAdvancedMode,
        #                       'Ctrl+Shift+A', 'expert', u'Switch to advanced mode',
        #                       checkable=True)

        hideAll = action('&Hide\nBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', u'Hide all Boxs',
                         enabled=False)
        showAll = action('&Show\nBox', partial(self.togglePolygons, True),
                         'Ctrl+Alt+A', 'hide', u'Show all Boxs',
                         enabled=False)

        help = action('&Tutorial', self.tutorial, 'Ctrl+T', 'help',
                      u'Show demos')

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action('Zoom &In', partial(self.addZoom, 5),
                        'Ctrl++', 'zoom-in', u'Increase zoom level', enabled=False)
        zoomOut = action('&Zoom Out', partial(self.addZoom, -5),
                         'Ctrl+-', 'zoom-out', u'Decrease zoom level', enabled=False)
        zoomOrg = action('&Original size', partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', u'Zoom to original size', enabled=False)
        fitWindow = action('&Fit Window', self.setFitWindow,
                           'Ctrl+F', 'fit-window', u'Zoom follows window size',
                           checkable=True, enabled=False)
        fitWidth = action('Fit &Width', self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', u'Zoom follows window width',
                          checkable=True, enabled=False)

        delete_label_file = action(
            self.tr("Delete\nLabel File"),
            self._delete_label_file,
            '',
            "delete_label_file",
            self.tr("Delete selected label file"),
        )
        delete_label_file.setIcon(QIcon("aligner_gui\\labeler\\icons\\delete_label_file.png"))

        save_selected.setIcon(QIcon("aligner_gui\\labeler\\icons\\save.png"))

        remove_selected_images = action(
            self.tr("Remove\nSelected"),
            self._remove_selected_images_from_list,
            'Ctrl+Delete',
            "remove_selected_images",
            self.tr("Remove selected images from the current project list."),
        )
        remove_selected_images.setIcon(QIcon("aligner_gui\\labeler\\icons\\delete_label_file.png"))

        auto_labeling = action(
            self.tr("Auto Labeling"),
            self._auto_label,
            '',
            "auto_labeling",
            self.tr("Label automatically with the trained model."),
        )
        auto_labeling.setIcon(QIcon("aligner_gui\\labeler\\icons\\049-magic-wand.png"))


        toggle_keep_prev_mode = action(
            self.tr("Keep Previous\nAnnotation"),
            self._toggle_keep_prev,
            '',
            "keep_prev_annot",
            self.tr('Toggle "keep previous annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setIcon(QIcon("aligner_gui\\labeler\\icons\\keep_prev_annot.png"))
        self._is_keep_prev = toggle_keep_prev_mode.isChecked()
        toggle_keep_prev_mode.setChecked(self._is_keep_prev)

        kept_box_infos = KeptBoxAction(self)
        apply_kept_box_to_all = action(
            self.tr("Apply Kept box\nto all"),
            self._apply_kept_box_to_all,
            '',
            "apply_kept_box_to_all",
            self.tr("Apply kept box to all boxes having same class name which you select."
                    "\r\nIt only changes longside and shortside of box."),
        )
        apply_kept_box_to_all.setIcon(QIcon("aligner_gui\\labeler\\icons\\apply.png"))


        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action('&Edit Label', self._edit_label_name,
                      'Ctrl+E', 'edit', u'Modify the label of the selected Box',
                      enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText('Show/Hide Label Panel')
        labels.setShortcut('Ctrl+Shift+L')

        # Lavel list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))
        self._label_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self._label_list_widget.customContextMenuRequested.connect(
            self.popLabelListMenu)

        fileListMenu = QMenu()
        addActions(fileListMenu, (save_selected, delete_label_file, remove_selected_images))
        self._file_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self._file_list_widget.customContextMenuRequested.connect(
            self.popFileListMenu)

        # Store actions for further handling.
        self.actions = struct(open=open, close=close,
                              delete=delete, edit=edit, copy=copy, paste=paste, keep=keep, apply=apply,
                              createMode=createMode, editMode=editMode,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              fileMenuActions=(
                                  open, opendir, close, quit),
                              editMenu=(
                                  edit, copy, paste, delete),
                              advancedContext=(
                                  keep, apply, createMode, editMode, edit, copy, paste, delete),
                              onLoadActive=(
                                  close, createMode, editMode),
                              on_shapes_present=(hideAll, showAll),
                              toggle_keep_prev_mode=toggle_keep_prev_mode,
                              save_selected=save_selected,
                              delete_label_file=delete_label_file,
                              remove_selected_images=remove_selected_images,
                              auto_labeling=auto_labeling,
                              apply_kept_box_to_all=apply_kept_box_to_all,
                              kept_box_infos=kept_box_infos)

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu,
            fileList=fileListMenu)

        self.menuBar().setVisible(False)

        addActions(self.menus.file,
                   (open, opendir, self.menus.recentFiles, close, None, quit))
        addActions(self.menus.help, (help,))
        addActions(self.menus.view, (
            labels, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.advancedContext)

        self.tools = self.toolbar('Tools')

        self.actions.advanced = (
            opendir, openPrevImg, openNextImg, None, createMode, editMode, None, copy, paste, None,
            save, save_selected, delete, delete_label_file, remove_selected_images, None, toggle_keep_prev_mode, None, auto_labeling, None,
            apply_kept_box_to_all, kept_box_infos, None)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self._current_image = QImage()
        self._current_image_path: str = None
        self.recentFiles = []
        self.maxRecent = 7

        self.zoom_level = 100
        self.fit_window = False
        self._select_all_files_shortcut = QShortcut(QKeySequence("Alt+A"), self)
        self._select_all_files_shortcut.setContext(Qt.WindowShortcut)
        self._select_all_files_shortcut.activated.connect(self._select_all_files_in_list)
        self._file_list_select_all_shortcut = QShortcut(QKeySequence.SelectAll, self._file_list_widget)
        self._file_list_select_all_shortcut.activated.connect(self._select_all_files_in_list)
        self._save_selected_shortcut = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
        self._save_selected_shortcut.setContext(Qt.WindowShortcut)
        self._save_selected_shortcut.activated.connect(self._save_selected_labels)
        # Add Chris
        self.difficult = False

        self.recentFiles = []
        size = QSize(600, 500)
        position = QPoint(0, 0)
        self.resize(size)
        self.move(position)
        self.lastOpenDir = None

        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(QByteArray())

        # Add chris
        Shape.difficult = self.difficult

        self.canvas.setEditing(True)
        self.populateModeActions()
        self.actions.createMode.setEnabled(True)
        self.actions.editMode.setEnabled(False)

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        # initialize image_paths
        if self._is_new == True:
            self._set_labeler_image_list_to_file([])

        image_paths = self._get_labeler_image_list_from_file()
        self._load_images(image_paths)

    @staticmethod
    def _get_label_dialog_class():
        from aligner_gui.labeler.libs.labelDialog import LabelDialog

        return LabelDialog

    @staticmethod
    def _get_select_class_name_dialog_class():
        from aligner_gui.labeler.libs.select_class_name_dlg import SelectClassNameDialog

        return SelectClassNameDialog

    @staticmethod
    def _get_label_file_class():
        from aligner_gui.labeler.libs.labelFile import LabelFile

        return LabelFile

    @staticmethod
    def _get_progress_list_dialog_class():
        from aligner_gui.widgets.progress_list_dialog import ProgressListDialog

        return ProgressListDialog

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar

    ## Support Functions ##

    def no_shapes(self):
        return not self._label_list_items_to_shapes

    def populateModeActions(self):
        tool = self.actions.advanced
        menu = self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def _set_dirty(self):  # It means there are someting to save
        self._dirty = True
        if self._autoSaving == True:
            self._autosave_timer.start()

    def _set_clean(self):
        self._dirty = False
        self._autosave_timer.stop()

    def _flush_autosave(self):
        if self._dirty and self._current_image_path is not None:
            self._save_label()

    def enableCreate(self, b):
        self.isEnableCreate = not b
        self.actions.create.setEnabled(self.isEnableCreate)

    def enableCreateRo(self, b):
        self.isEnableCreateRo = not b
        self.actions.createRo.setEnabled(self.isEnableCreateRo)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        # print(message)
        self.statusBar().showMessage(message, delay)
        self.statusBar().show()

    def resetState(self):
        self._label_list_items_to_shapes.clear()
        self._shapes_to_label_list_items.clear()
        self._label_list_widget.clear()
        self._current_image_path = None
        self.canvas.resetState()

    def _label_list_current_item(self):
        items = self._label_list_widget.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    ## Callbacks ##
    def tutorial(self):
        subprocess.Popen([self.screencastViewer, self.screencast])

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        self.toggleDrawMode(False)

    def setEditMode(self):
        self.toggleDrawMode(True)

    def _keep_longside_shortside_of_box(self):
        long, short = self.canvas.get_longside_shortside_of_selected_box()
        self.actions.kept_box_infos.set_kept_box_info(long, short)

    def _apply_longside_shortside_of_kept_box(self):
        long, short = self.actions.kept_box_infos.get_kept_box_info()
        if long == 0 or short == 0:
            gui_util.get_message_box(self, "An error occurred while applying",
                                     "The long and short values of the kept box must be greater than 0!")
            self.actions.kept_box_infos.spinbox_with_value_zero_is_highlighted()
            return
        self._remove_shape(self.canvas.set_longside_shortside_of_selected_box(long, short))
        self._set_dirty()
        if self.no_shapes():
            for action in self.actions.on_shapes_present:
                action.setEnabled(False)


    def updateFileMenu(self):
        currFilePath = self._current_image_path

        def exists(filename):
            return os.path.exists(filename)

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self._label_list_widget.mapToGlobal(point))

    def popFileListMenu(self, point):
        self.menus.fileList.exec_(self._file_list_widget.mapToGlobal(point))

    def _edit_label_name(self, item=None):
        if not self.canvas.editing():
            return
        item = item if item else self._label_list_current_item()
        LabelDialog = self._get_label_dialog_class()
        label_dlg = LabelDialog(parent=self, label_names=LabelManager.get_label_names_with_idx())
        text = label_dlg.popUp(item.text())
        if text is not None:
            item.setText(text)
            self._set_dirty()
            LabelManager.update_label_names_with_idx(text)

    def _apply_kept_box_to_all(self):
        long, short = self.actions.kept_box_infos.get_kept_box_info()
        if long == 0 or short == 0:
            gui_util.get_message_box(self, "An error occurred while to all",
                                     "The long and short values of the kept box must be greater than 0!")
            self.actions.kept_box_infos.spinbox_with_value_zero_is_highlighted()
            return

        SelectClassNameDialog = self._get_select_class_name_dialog_class()
        select_class_name_dlg = SelectClassNameDialog(parent=self, label_names=LabelManager.get_label_names_with_idx())
        text = select_class_name_dlg.pop_up()

        if text is None:
            return

        img_paths = []
        for img_path, is_labeled in self._labeled_info_dict.items():
            if is_labeled:
                img_paths.append(img_path)

        def work(idx: int):
            try:
                img_path = img_paths[idx]
                LabelFile = self._get_label_file_class()
                label_file = LabelFile(img_path)
                label_file.load_label()
                shapes = []
                is_need_confirm = False
                for shape in label_file.get_shapes():
                    if shape.get_label() == text:
                        shape, valid = self.canvas.set_longside_shortside_of_shape_box(shape, long, short)
                        if valid == True:
                            if shape is not None:
                                shapes.append(shape)
                            is_need_confirm = True
                        else:
                            shapes.append(shape)
                    else:
                        shapes.append(shape)

                if is_need_confirm:
                    if len(shapes) > 0:
                        current_image = QImage.fromData(util.read_file(ustr(img_path), None))
                        label_file.set_shapes(shapes)
                        label_file.save_label(image_info={
                            "height": current_image.height(),
                            "width": current_image.width(),
                            "depth": 1 if current_image.isGrayscale() else 3,
                            "isNeedConfirm": True,
                        })
                    else:
                        label_file.remove_label_file()

            except Exception as e:
                traceback.print_tb(e.__traceback__)
                error_msg = "ERROR - " + str(e)
                logging.error(error_msg)
                return

        ProgressListDialog = self._get_progress_list_dialog_class()
        dlg = ProgressListDialog(work, img_paths)
        dlg.exec_()

        gui_util.get_message_box(self, "Apply kept box to all", "Appling kept box to all label files was successfully.")
        image_paths = self._get_labeler_image_list_from_file()
        self._load_images(image_paths)



    # Tzutalin 20160906 : Add file list and dock to move faster
    def _file_list_item_clicked(self, item=None):
        if item is None:
            return
        modifiers = QApplication.keyboardModifiers()
        if modifiers & (Qt.ControlModifier | Qt.ShiftModifier):
            return
        if len(self._file_list_widget.selectedItems()) > 1:
            return
        target_path = ustr(item.text())
        if target_path == self._current_image_path:
            return
        currIndex = self._image_paths.index(target_path)
        if currIndex < len(self._image_paths):
            filename = self._image_paths[currIndex]
            if filename:
                self._load_file(filename)

    # React to canvas signals.
    def _canvas_shape_selection_changed(self, selected=False):
        label_info = ""
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selected_shape
            if shape:
                self._shapes_to_label_list_items[shape].setSelected(True)

                point_0 = [shape.points[0].x(), shape.points[0].y()]
                point_1 = [shape.points[1].x(), shape.points[1].y()]
                point_2 = [shape.points[2].x(), shape.points[2].y()]

                side_0 = math.dist(point_0, point_1)
                side_1 = math.dist(point_1, point_2)
                if side_0 >= side_1:
                    long = side_0
                    short = side_1
                else:
                    long = side_1
                    short = side_0

                label_info = 'L: {0}, Long: {1:.1f}, Short: {2:.1f}'.format(shape.get_label(), long,
                                                                            short)
            else:
                self._label_list_widget.clearSelection()

        self._lbl_label_info.setText(label_info)
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.keep.setEnabled(selected)
        self.actions.apply.setEnabled(selected)
        self.actions.edit.setEnabled(selected)

    def _add_shape(self, shape: Shape):
        item = HashableQListWidgetItem(shape.get_label())
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self._label_list_items_to_shapes[item] = shape
        self._shapes_to_label_list_items[shape] = item
        self._label_list_widget.addItem(item)
        for action in self.actions.on_shapes_present:
            action.setEnabled(True)

    def _remove_shape(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self._shapes_to_label_list_items[shape]
        self._label_list_widget.takeItem(self._label_list_widget.row(item))
        del self._shapes_to_label_list_items[shape]
        del self._label_list_items_to_shapes[item]

    def _copy_selected_shape(self):
        self._copied_shape = deepcopy(self.canvas.selected_shape)

    def _paste_copied_shape(self):
        if self._copied_shape == None:
            return
        copied_shape = deepcopy(self._copied_shape)
        self._add_shape(copied_shape)
        self.canvas.append_shape(copied_shape)
        self._set_dirty()
        # self._canvas_shape_selection_changed(True)

    def _label_list_selection_changed(self):
        item = self._label_list_current_item()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self._label_list_items_to_shapes[item])
            shape = self._label_list_items_to_shapes[item]

    def _label_list_name_edited(self, item):
        shape = self._label_list_items_to_shapes[item]
        label = item.text()
        if label != shape.get_label():
            shape.set_label(item.text())
            self._set_dirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def _new_shape_made_from_canvas(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """

        LabelDialog = self._get_label_dialog_class()
        label_dlg = LabelDialog(parent=self, label_names=LabelManager.get_label_names_with_idx())

        text = label_dlg.popUp(text=self._prevLabelText)
        if text is not None:
            self._prevLabelText = text
            self._add_shape(self.canvas.setLastLabel(text))

            self.actions.editMode.setEnabled(True)
            self._set_dirty()
            LabelManager.update_label_names_with_idx(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def moveViewPointPressed(self):
        self.moveViewPointPressedScrollHorizontal = self.scrollBars[Qt.Horizontal].value()
        self.moveViewPointPressedScrollVetical = self.scrollBars[Qt.Vertical].value()

    def moveViewPointHorizontalRequest(self, delta_horizontal):
        value = self.moveViewPointPressedScrollHorizontal + (delta_horizontal)
        value = min(value, self.scrollBars[Qt.Horizontal].maximum())
        value = max(value, 0)
        self.setScroll(Qt.Horizontal, value)

    def moveViewPointVerticalRequest(self, delta_vertical):
        value = self.moveViewPointPressedScrollVetical + (delta_vertical)
        value = min(value, self.scrollBars[Qt.Vertical].maximum())
        value = max(value, 0)
        self.setScroll(Qt.Vertical, value)


    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = delta / (8 * 15)
        current_zoom = max(self.zoomWidget.value(), self.zoomWidget.minimum())
        target_zoom = int(round(current_zoom * (self.ZOOM_WHEEL_FACTOR ** units)))
        if target_zoom == current_zoom:
            target_zoom = current_zoom + (1 if units > 0 else -1)
        target_zoom = max(self.zoomWidget.minimum(), min(self.zoomWidget.maximum(), target_zoom))
        self.setZoom(target_zoom)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self._label_list_items_to_shapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def get_qimage_from_mat(self, mat_image):
        import cv2

        if mat_image is None:
            return QImage()
        if len(mat_image.shape) == 2:
            image_array = cv2.cvtColor(mat_image, cv2.COLOR_GRAY2RGB)
        elif mat_image.shape[2] == 4:
            image_array = cv2.cvtColor(mat_image, cv2.COLOR_BGRA2RGBA)
        else:
            image_array = cv2.cvtColor(mat_image, cv2.COLOR_BGR2RGB)

        height, width, channel = image_array.shape
        bytes_per_line = channel * width
        image_format = QImage.Format_RGBA8888 if channel == 4 else QImage.Format_RGB888
        qimage = QImage(image_array.data, width, height, bytes_per_line, image_format)
        return qimage.copy()

    def _read_image_mat(self, file_path):
        return self._image_reader.read(file_path)

    def _decode_image_mat(self, file_path):
        import cv2

        image = decode_image_with_cv2(file_path, cv2)
        if image is None:
            return None
        if len(image.shape) == 3 and image.shape[2] == 3:
            return image
        try:
            raw = np.fromfile(file_path, dtype=np.uint8)
            if raw.size == 0:
                return None
            return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        except Exception:
            return None

    def _load_file(self, file_path=None, is_load_after_delete=False):
        """Load the specified file, or the last opened file if None."""
        prev_shapes = self.canvas.get_shape()
        self.resetState()
        self.canvas.setEnabled(False)
        if file_path is None:
            file_path = self.settings.get('filename')

        unicodeFilePath = ustr(file_path)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self._file_list_widget.count() > 0:
            index = self._image_paths.index(unicodeFilePath)
            fileWidgetItem = self._file_list_widget.item(index)
            fileWidgetItem.setSelected(True)

        if unicodeFilePath and os.path.exists(unicodeFilePath):

            image = self.get_qimage_from_mat(self._read_image_mat(unicodeFilePath))
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self._current_image = image
            self._current_image_path = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            self._set_clean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self._current_image_path)

            LabelFile = self._get_label_file_class()
            label_file = LabelFile(self._current_image_path)
            label_file.load_label()
            current_shapes = label_file.get_shapes()
            shapes = current_shapes

            is_loaded_from_prev = False
            if self._is_keep_prev:
                if (len(current_shapes) == 0) and (not is_load_after_delete) and (len(prev_shapes) > 0):
                    shapes = prev_shapes
                    is_loaded_from_prev = True

            for shape in shapes:
                self._add_shape(shape)

            self.canvas.loadShapes(shapes)
            self._main_window.setWindowTitle(__appname__ + ' ' + self._current_image_path)

            # Default : select last item if there is at least one item
            if self._label_list_widget.count():
                self._label_list_widget.setCurrentItem(
                    self._label_list_widget.item(self._label_list_widget.count() - 1))
                # self.labelList.setItemSelected(self.labelList.item(self.labelList.count()-1), True)

            self.canvas.setFocus(True)
            if is_loaded_from_prev:
                self._set_dirty()

            return True
        return False

    def resizeEvent(self, event):
        if self.canvas and not self._current_image.isNull() \
                and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(LabelerWidget, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self._current_image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if self._image_index_thread is not None and self._image_index_thread.isRunning():
            self._image_index_thread.request_cancel()
            self._image_index_thread.wait(3000)
        if self._image_index_progress is not None:
            self._image_index_progress.close()
            self._image_index_progress.deleteLater()
            self._image_index_progress = None
        if not self.mayContinue():
            event.ignore()

    ## User Dialogs ##

    def loadRecent(self, filename):
        if self.mayContinue():
            self._load_file(filename)

    def scanAllImages(self, folderPath):
        extensions = gui_util.SUPPORTED_IMAGE_FORMATS
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relatviePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relatviePath))
                    images.append(path)
        images.sort(key=lambda x: x.lower())
        return images

    def _open_dir(self, _value=False):
        if not self.mayContinue():
            return

        path = os.path.dirname(self._current_image_path) \
            if self._current_image_path else '.'

        if self.lastOpenDir is not None and len(self.lastOpenDir) > 1:
            path = self.lastOpenDir

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                        '%s - Open Directory' % __appname__, path,
                                                        QFileDialog.ShowDirsOnly
                                                        | QFileDialog.DontResolveSymlinks))

        if dirpath == "":
            return

        self.lastOpenDir = dirpath

        self._dir_name = dirpath
        self._current_image_path = None

        image_paths = self.scanAllImages(dirpath)
        self._set_labeler_image_list_to_file(image_paths)
        self._load_images(image_paths)

    def _set_labeler_image_list_to_file(self, image_paths):
        try:
            save_labeler_image_list(self._project_path, image_paths)
        except Exception as e:
            print(e)

    def _get_labeler_image_list_from_file(self):
        try:
            return load_labeler_image_list(self._project_path)
        except Exception as e:
            return []

    def _load_images(self, img_paths):
        self._image_reader.clear()
        if self._image_index_thread is not None and self._image_index_thread.isRunning():
            self._image_index_thread.request_cancel()
            self._image_index_thread.wait(3000)
        if self._image_index_progress is not None:
            self._image_index_progress.close()
            self._image_index_progress.deleteLater()
            self._image_index_progress = None

        if len(img_paths) == 0:
            self._apply_image_states([])
            return

        self._file_list_widget.setEnabled(False)
        self.status("Scanning image labels...")

        self._image_index_thread = ImageIndexThread(img_paths)
        self._image_index_thread.sig_progress.connect(self._progress_image_index)
        self._image_index_thread.sig_completed.connect(self._completed_image_index)
        self._image_index_thread.sig_failed.connect(self._failed_image_index)

        if self.isVisible():
            self._image_index_progress = QProgressDialog("Scanning image labels...", "Cancel", 0, len(img_paths), self)
            self._image_index_progress.setWindowTitle("Indexing Images")
            self._image_index_progress.setWindowModality(Qt.WindowModal)
            self._image_index_progress.setMinimumDuration(0)
            self._image_index_progress.setValue(0)
            self._image_index_progress.canceled.connect(self._image_index_thread.request_cancel)
            self._image_index_progress.show()
        else:
            self._image_index_progress = None

        self._image_index_thread.start()

    def _progress_image_index(self, cur_idx: int, total: int, path: str):
        progress = self._image_index_progress
        if progress is not None:
            try:
                progress.setMaximum(max(total, 1))
                progress.setLabelText(f"Scanning image labels...\n{os.path.basename(path)}")
                progress.setValue(cur_idx)
            except RuntimeError:
                self._image_index_progress = None
        self.status(f"Scanning image labels... ({cur_idx}/{total})", 0)

    def _completed_image_index(self, states, was_cancelled: bool):
        if self._image_index_progress is not None:
            self._image_index_progress.close()
            self._image_index_progress.deleteLater()
            self._image_index_progress = None

        self._image_index_thread = None
        if was_cancelled:
            self._file_list_widget.setEnabled(True)
            self.status("Image indexing canceled.", 3000)
            return

        self._apply_image_states(states)

    def _failed_image_index(self, message: str):
        if self._image_index_progress is not None:
            self._image_index_progress.close()
            self._image_index_progress.deleteLater()
            self._image_index_progress = None

        self._image_index_thread = None
        self._file_list_widget.setEnabled(True)
        logging.error("ERROR - %s", message)
        gui_util.get_message_box(self, "Indexing Failed", "Failed to scan image labels.")
        self.status("Image indexing failed.", 3000)

    def _apply_image_states(self, states):
        LabelManager.init()
        self._file_list_widget.clear()

        for state in states:
            for label_name in state.labels:
                LabelManager.update_label_names_with_idx(label_name)

        self._image_paths = [state.path for state in states]
        self._file_list_total_count = len(self._image_paths)
        self._file_list_labeled_count = 0
        self._labeled_info_dict.clear()
        self._pending_image_states = list(states)
        self._pending_image_state_index = 0
        if len(self._pending_image_states) == 0:
            self._finish_apply_image_states()
            return

        self.status("Preparing image list...", 0)
        QTimer.singleShot(0, self._append_next_image_state_batch)

    def _append_next_image_state_batch(self):
        total = len(self._pending_image_states)
        if total == 0:
            self._finish_apply_image_states()
            return

        start = self._pending_image_state_index
        end = min(start + self.IMAGE_LIST_BATCH_SIZE, total)

        self._file_list_widget.setUpdatesEnabled(False)
        for state in self._pending_image_states[start:end]:
            item = QListWidgetItem(state.path)
            self._set_file_item_state(item, state.path, state.has_label, state.is_empty, state.needs_confirm)
            if state.has_label:
                self._file_list_labeled_count += 1
            self._file_list_widget.addItem(item)
        self._file_list_widget.setUpdatesEnabled(True)

        self._pending_image_state_index = end
        self.status(f"Preparing image list... ({end}/{total})", 0)
        if end < total:
            QTimer.singleShot(0, self._append_next_image_state_batch)
            return

        self._finish_apply_image_states()

    def _finish_apply_image_states(self):
        self._pending_image_states = []
        self._pending_image_state_index = 0
        self._file_list_widget.setEnabled(True)
        self._refresh_lbl_file_list_info()
        self.status("Image indexing finished.", 3000)
        self.queueEvent(self.openNextImg)

    def openPrevImg(self, _value=False):
        if not self.mayContinue():
            return

        if len(self._image_paths) <= 0:
            return

        if self._current_image_path is None:
            return

        currIndex = self._image_paths.index(self._current_image_path)
        if currIndex - 1 >= 0:
            filename = self._image_paths[currIndex - 1]
            if filename:
                self._load_file(filename)

    def openNextImg(self, _value=False):
        if not self.mayContinue():
            return

        if len(self._image_paths) <= 0:
            return

        filename = None
        if self._current_image_path is None:
            filename = self._image_paths[0]
        else:
            currIndex = self._image_paths.index(self._current_image_path)
            if currIndex + 1 < len(self._image_paths):
                filename = self._image_paths[currIndex + 1]

        if filename:
            self._load_file(filename)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self._current_image_path)) if self._current_image_path else '.'
        formats = gui_util.SUPPORTED_IMAGE_FORMATS
        LabelFile = self._get_label_file_class()
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.SUFFIX])
        filename = QFileDialog.getOpenFileName(self,
                                               '%s - Choose Image or Label file' % __appname__,
                                               path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self._load_file(filename)

    def _save_label(self, _value=False):
        if self._current_image_path is None or self._current_image.isNull():
            return
        LabelFile = self._get_label_file_class()
        label_file = LabelFile(self._current_image_path)
        label_file.set_shapes(self.canvas.get_shape())
        label_file.save_label(image_info={
            "height": self._current_image.height(),
            "width": self._current_image.width(),
            "depth": 1 if self._current_image.isGrayscale() else 3,
            "isNeedConfirm": False,
        })

        # post process
        self._set_clean()
        current_shapes = label_file.get_shapes()
        self._mark_path_as_saved(
            self._current_image_path,
            has_label=True,
            is_empty=(len(current_shapes) == 0),
            needs_confirm=False,
        )

    def _save_selected_labels(self, _value=False):
        target_paths = self._get_selected_image_paths()
        if not target_paths or self._current_image_path is None:
            return

        selected_shapes = deepcopy(self.canvas.get_shape())
        saved_paths = []
        failed_paths = []
        if len(target_paths) > 1:
            msg = self.tr(
                "Save current labels to {} selected images?\r\n"
                "Existing label files of those images will be overwritten."
            ).format(len(target_paths))
            if QMessageBox.warning(self, self.tr("Attention"), msg, QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return

        def work(idx: int):
            target_path = target_paths[idx]
            try:
                self._save_shapes_to_image_path(target_path, selected_shapes)
                saved_paths.append(target_path)
            except Exception as e:
                logging.exception("Failed to save labels to %s", target_path)
                failed_paths.append(target_path)

        ProgressListDialog = self._get_progress_list_dialog_class()
        dlg = ProgressListDialog(work, target_paths)
        dlg.exec_()

        for target_path in saved_paths:
            self._mark_path_as_saved(
                target_path,
                has_label=True,
                is_empty=(len(selected_shapes) == 0),
                needs_confirm=False,
            )
        if self._current_image_path in saved_paths:
            self._set_clean()
        if failed_paths:
            gui_util.get_message_box(
                self,
                "Batch Save",
                f"Saved {len(saved_paths)} image(s).\nFailed: {len(failed_paths)} image(s). Check the log for details.",
            )
        self.status(f"Saved labels to {len(saved_paths)} image(s).", 3000)

    def saveFileDialog(self):
        caption = '%s - Choose File' % __appname__
        LabelFile = self._get_label_file_class()
        filters = 'File (*%s)' % LabelFile.SUFFIX
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.SUFFIX[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self._current_image_path)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            return dlg.selectedFiles()[0]
        return ''

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self._set_clean()
        self.canvas.setEnabled(False)

    def mayContinue(self):
        if self._dirty and self._autoSaving:
            self._flush_autosave()
        return not (self._dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'You have unsaved changes, proceed anyway?'
        return yes == QMessageBox.warning(self, u'Attention', msg, yes | no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self._current_image_path) if self._current_image_path else '.'

    def _delete_selected_shape(self):
        self._remove_shape(self.canvas.delete_selected())
        self._set_dirty()
        if self.no_shapes():
            for action in self.actions.on_shapes_present:
                action.setEnabled(False)

    def make_dataset_summary(self, dataset_summary_path, include_empty:bool):
        return build_dataset_summary(self._image_paths, dataset_summary_path, include_empty)

    def _toggle_keep_prev(self):
        self._is_keep_prev = not self._is_keep_prev

    def _delete_label_file(self):
        selected_items = self._file_list_widget.selectedItems()
        target_paths = [ustr(item.text()) for item in selected_items]
        if not target_paths and self._current_image_path is not None:
            target_paths = [self._current_image_path]
        if not target_paths:
            return

        mb = QMessageBox
        if len(target_paths) == 1:
            msg = self.tr(
                "You are about to permanently delete label file of {}, \r\n"
                "proceed anyway?"
            ).format(target_paths[0])
        else:
            msg = self.tr(
                "You are about to permanently delete {} label files, \r\n"
                "proceed anyway?"
            ).format(len(target_paths))
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        LabelFile = self._get_label_file_class()
        current_image_path = self._current_image_path
        current_deleted = current_image_path in target_paths
        for target_path in target_paths:
            label_file = LabelFile(target_path)
            label_file.remove_label_file()
            if self._labeled_info_dict.get(target_path, False):
                self._labeled_info_dict[target_path] = False
                self._file_list_labeled_count = self._file_list_labeled_count - 1
            self._set_file_item_state_by_path(target_path, has_label=False)

        self._refresh_lbl_file_list_info()
        if current_deleted and current_image_path is not None:
            self.resetState()
            self._load_file(current_image_path, is_load_after_delete=True)

    def _remove_selected_images_from_list(self):
        selected_items = self._file_list_widget.selectedItems()
        target_paths = [ustr(item.text()) for item in selected_items]
        if not target_paths:
            return

        mb = QMessageBox
        if len(target_paths) == 1:
            msg = self.tr(
                "You are about to remove {} from the current project list.\r\n"
                "The image file itself will not be deleted. Proceed anyway?"
            ).format(target_paths[0])
        else:
            msg = self.tr(
                "You are about to remove {} images from the current project list.\r\n"
                "The image files themselves will not be deleted. Proceed anyway?"
            ).format(len(target_paths))
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        removed_paths = set(target_paths)
        current_path = self._current_image_path
        removal_result = remove_paths_from_file_list(
            self._image_paths,
            self._labeled_info_dict,
            removed_paths,
            current_path,
        )

        for row in range(self._file_list_widget.count() - 1, -1, -1):
            item = self._file_list_widget.item(row)
            item_path = ustr(item.text())
            if item_path not in removed_paths:
                continue
            self._file_list_widget.takeItem(row)

        self._image_paths = removal_result.image_paths
        self._labeled_info_dict = removal_result.labeled_info
        self._file_list_labeled_count = removal_result.labeled_count
        self._file_list_total_count = len(removal_result.image_paths)
        self._set_labeler_image_list_to_file(self._image_paths)
        self._refresh_lbl_file_list_info()

        if removal_result.removed_current:
            self.resetState()
            self._set_clean()
            self.canvas.setEnabled(False)
            self._current_image_path = None
            self._current_image = QImage()
            if removal_result.next_image_path is not None:
                self._load_file(removal_result.next_image_path)
            else:
                self._main_window.setWindowTitle(__appname__)

        self.status(f"Removed {removal_result.removed_count} image(s) from the project list.", 3000)

    def _auto_label(self):
        import torch
        from aligner_engine.detector import Detector

        def join_path(*paths):
            osp_path = os.path.join(*paths)
            return osp_path.replace('\\', '/')

        if not self._worker.is_there_trained_checkpoint():
            gui_util.get_message_box(self, "Invalid Export", "There is no trained model. "
                                                             "To use Auto Labeling function, You have to train a model with some labeled data before.")
            return

        detector = None
        try:
            config_path_src = join_path(self._project_path, const.DIRNAME_AUTOSAVED,
                                        const.FILENAME_MODEL_CONFIG)
            config_path_dst = join_path(self._project_path, "auto_labeling_" + const.FILENAME_MODEL_CONFIG)
            copyfile(config_path_src, config_path_dst)

            ckpt_path_src = join_path(self._project_path, const.DIRNAME_AUTOSAVED,
                                      const.FILENAME_CKPT)
            ckpt_path_dst = join_path(self._project_path, "auto_labeling_" + const.FILENAME_CKPT)
            copyfile(ckpt_path_src, ckpt_path_dst)

            device = "cuda:0"
            if not torch.cuda.is_available():
                device = "cpu"

            detector = Detector(config_path_dst, ckpt_path_dst, device)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            error_msg = "ERROR - " + str(e)
            logging.error(error_msg)
            return

        img_paths = []
        for img_path, is_labeled in self._labeled_info_dict.items():
            if not is_labeled:
                img_paths.append(img_path)

        def work(idx: int):
            try:
                img_path = img_paths[idx]
                img = self._read_image_mat(img_path)
                if img is None:
                    return
                result = detector.inference(img)

                if len(result) > 0:
                    LabelFile = self._get_label_file_class()
                    label_file = LabelFile(img_path)
                    shapes = []
                    for box_idx, box_info in result.items():
                        if box_info['conf'] >= 0.5:
                            shape = Shape(label=box_info['class_name'])
                            shape.addPoint(QPointF(box_info['qbox'][0], box_info['qbox'][1]))
                            shape.addPoint(QPointF(box_info['qbox'][2], box_info['qbox'][3]))
                            shape.addPoint(QPointF(box_info['qbox'][4], box_info['qbox'][5]))
                            shape.addPoint(QPointF(box_info['qbox'][6], box_info['qbox'][7]))
                            shape.isRotated = True
                            shape.close()
                            shapes.append(shape)
                    if len(shapes) > 0:
                        current_image = QImage.fromData(util.read_file(ustr(img_path), None))
                        label_file.set_shapes(shapes)
                        label_file.save_label(image_info={
                            "height": current_image.height(),
                            "width": current_image.width(),
                            "depth": 1 if current_image.isGrayscale() else 3,
                            "isNeedConfirm": True,
                        })
            except Exception as e:
                traceback.print_tb(e.__traceback__)
                error_msg = "ERROR - " + str(e)
                logging.error(error_msg)
                return

        ProgressListDialog = self._get_progress_list_dialog_class()
        dlg = ProgressListDialog(work, img_paths)
        dlg.exec_()

        if detector is not None:
            detector.to_cpu()
            del detector
            detector = None

        gui_util.get_message_box(self, "Auto Labeling", "Auto Labeling finished successfully.")
        image_paths = self._get_labeler_image_list_from_file()
        self._load_images(image_paths)

    def _item_selected_changed_file_list(self):
        indexes = self._file_list_widget.selectedIndexes()
        if len(indexes) > 0:
            item = self._file_list_widget.item(indexes[0].row())
            self._file_list_widget.scrollToItem(item)

    def _refresh_lbl_file_list_info(self):
        file_list_info = 'Total: {0}, Labeled: {1}'.format(self._file_list_total_count, self._file_list_labeled_count)
        self._lbl_file_list_info.setText(file_list_info)

    def _get_selected_image_paths(self):
        selected_items = self._file_list_widget.selectedItems()
        target_paths = [ustr(item.text()) for item in selected_items]
        if not target_paths and self._current_image_path is not None:
            target_paths = [self._current_image_path]
        return list(dict.fromkeys(target_paths))

    def _select_all_files_in_list(self):
        if self._file_list_widget.count() == 0:
            return
        self._file_list_widget.setFocus(Qt.ShortcutFocusReason)
        self._file_list_widget.selectAll()
        self.status(f"Selected {self._file_list_widget.count()} images.", 2000)

    def _set_file_item_state(self, item: QListWidgetItem, image_path: str, has_label: bool, is_empty: bool = False,
                             needs_confirm: bool = False):
        self._labeled_info_dict[image_path] = has_label
        if not has_label:
            foreground = QColor(235, 92, 92)
            background = QColor(62, 24, 28)
        elif is_empty:
            foreground = QColor(160, 166, 176)
            background = QColor(38, 42, 48)
        elif needs_confirm:
            foreground = QColor(255, 176, 245)
            background = QColor(54, 34, 60)
        else:
            foreground = QColor(232, 236, 241)
            background = QColor(30, 34, 38)
        item.setForeground(foreground)
        item.setBackground(background)

    def _set_file_item_state_by_path(self, image_path: str, has_label: bool, is_empty: bool = False,
                                     needs_confirm: bool = False):
        for item in self._file_list_widget.findItems(image_path, Qt.MatchExactly):
            self._set_file_item_state(item, image_path, has_label, is_empty, needs_confirm)

    def _mark_path_as_saved(self, image_path: str, has_label: bool, is_empty: bool = False, needs_confirm: bool = False):
        was_labeled = self._labeled_info_dict.get(image_path, False)
        if has_label and not was_labeled:
            self._file_list_labeled_count += 1
        elif not has_label and was_labeled:
            self._file_list_labeled_count = max(0, self._file_list_labeled_count - 1)
        self._set_file_item_state_by_path(image_path, has_label, is_empty, needs_confirm)
        self._refresh_lbl_file_list_info()

    def _save_shapes_to_image_path(self, image_path: str, shapes: List[Shape]):
        LabelFile = self._get_label_file_class()
        label_file = LabelFile(image_path)
        label_file.set_shapes(deepcopy(shapes))
        if image_path == self._current_image_path and not self._current_image.isNull():
            current_image = self._current_image
        else:
            current_image = self.get_qimage_from_mat(self._read_image_mat(image_path))
        if current_image.isNull():
            raise RuntimeError(f"Failed to read image: {image_path}")
        label_file.save_label(image_info={
            "height": current_image.height(),
            "width": current_image.width(),
            "depth": 1 if current_image.isGrayscale() else 3,
            "isNeedConfirm": False,
        })


class ElideLeftDelegate(QStyledItemDelegate):
    def __init__(self):
        super().__init__()

    def paint(self, painter, option, index):
        painter.save()
        value = index.data(Qt.DisplayRole)
        textcolor = index.model().data(index, Qt.TextColorRole)
        pen = QPen(option.palette.text().color() if textcolor == None else textcolor.color())
        painter.setPen(pen)
        if int(option.state & QStyle.State_Selected) > 0:
            painter.fillRect(option.rect, QColor(20, 100, 160))
        else:
            backgroundColor = index.model().data(index, Qt.BackgroundColorRole)
            if backgroundColor:
                painter.fillRect(option.rect, backgroundColor.color())
        painter.drawText(option.rect, Qt.AlignLeft | Qt.AlignVCenter,
                         option.fontMetrics.elidedText(str(value), Qt.ElideLeft,
                                                       option.rect.width()))

        painter.restore()

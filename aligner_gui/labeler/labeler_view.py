#!/usr/bin/env python
# -*- coding: utf8 -*-
from __future__ import annotations

import codecs
import os.path
import re
import subprocess
import time
from functools import partial
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import json
from typing import List


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
from aligner_gui.viewmodels.labeler_viewmodel import LabelerViewModel
from aligner_gui.labeler import utils as util
from aligner_gui.project.project_dataset_service import (
    build_dataset_summary,
    inspect_image_labels,
)
from aligner_gui.shared.image_cache import CachedImageReader, decode_image_with_cv2
import logging
import math
from copy import deepcopy

# auto labeling
import aligner_engine.const as const
from shutil import copyfile
import traceback
from aligner_gui.shared import gui_util


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


class LabelerView(QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))
    ZOOM_WHEEL_FACTOR = 1.05

    def __init__(self, main_window, session, project_path, is_new: bool):
        super(LabelerView, self).__init__()
        self.setWindowTitle(__appname__)
        # For loading all image under a directory
        from aligner_gui.main_window import MainWindow
        self._main_window: MainWindow = main_window
        self._dir_name = None
        self._project_path: str = project_path
        self._worker = session
        self._image_index_progress: QProgressDialog | None = None
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
        # Last known label-state per path (for re-rendering file list items)
        self._file_item_states: dict[str, tuple] = {}  # path -> (has_label, is_empty, needs_confirm)

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

        self._file_list_widget = _FileListWidget()
        self._file_list_widget.setUniformItemSizes(True)
        self._file_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._file_list_widget.setItemDelegate(ElideLeftDelegate())
        self._file_list_widget.itemDoubleClicked.connect(self._file_list_item_clicked)
        self._file_list_widget.itemClicked.connect(self._file_list_item_clicked)
        self._file_list_widget.itemSelectionChanged.connect(self._item_selected_changed_file_list)

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
        addActions(fileListMenu, (delete_label_file, remove_selected_images))
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
            save, delete, delete_label_file, remove_selected_images, None, toggle_keep_prev_mode, None, auto_labeling, None,
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
        # Note: Ctrl+Shift+S is already handled by the save_selected QAction shortcut;
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

        # initialize viewmodel and wire signals
        self.viewmodel = LabelerViewModel(
            ImageIndexThread, self._project_path, session=self._worker, parent=self
        )
        self.viewmodel.status_message_requested.connect(self.status)

        # image indexing lifecycle
        self.viewmodel.image_index_started.connect(self._on_image_index_started)
        self.viewmodel.image_index_progress.connect(self._on_image_index_progress)
        self.viewmodel.image_index_completed.connect(self._on_image_index_completed)
        self.viewmodel.image_index_failed.connect(self._on_image_index_failed)

        # file list state
        self.viewmodel.file_list_cleared.connect(self._on_file_list_cleared)
        self.viewmodel.file_list_batch_ready.connect(self._on_file_list_batch_ready)
        self.viewmodel.file_list_enabled_changed.connect(self._file_list_widget.setEnabled)
        self.viewmodel.file_list_item_state_changed.connect(self._on_file_list_item_state_changed)
        self.viewmodel.file_list_items_removed.connect(self._on_file_list_items_removed)
        self.viewmodel.file_list_info_updated.connect(self._on_file_list_info_updated)

        # navigation
        self.viewmodel.navigate_to_image.connect(self._load_file)
        self.viewmodel.navigate_reset.connect(self._on_navigate_reset)

        # label state
        self.viewmodel.label_saved.connect(self._set_clean)
        self.viewmodel.label_files_deleted.connect(self._on_label_files_deleted)

        # image reader
        self.viewmodel.image_reader_clear_requested.connect(self._image_reader.clear)

        if self._is_new:
            self.viewmodel.save_labeler_image_list([])

        image_paths = self.viewmodel.get_labeler_image_list()
        image_paths.sort(
            key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', x)]
        )
        if image_paths:
            self.viewmodel.save_labeler_image_list(image_paths)
        self.viewmodel.load_images(image_paths)

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
        from aligner_gui.shared.progress_list_dialog import ProgressListDialog

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
        if self._autoSaving:
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

        img_paths = self.viewmodel.get_labeled_image_paths()

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
        image_paths = self.viewmodel.get_labeler_image_list()
        self.viewmodel.load_images(image_paths)



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
        image_paths = self.viewmodel.get_image_paths()
        currIndex = image_paths.index(target_path)
        if currIndex < len(image_paths):
            filename = image_paths[currIndex]
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
        if getattr(self, '_loading_file', False):
            return False  # Prevent re-entrant calls (e.g. from setSelected inside this method)
        self._loading_file = True
        try:
            return self._load_file_impl(file_path, is_load_after_delete)
        finally:
            self._loading_file = False

    def _load_file_impl(self, file_path=None, is_load_after_delete=False):
        prev_shapes = self.canvas.get_shape()
        self.resetState()
        self.canvas.setEnabled(False)
        if file_path is None:
            file_path = self.settings.get('filename')

        unicodeFilePath = ustr(file_path)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self._file_list_widget.count() > 0:
            image_paths = self.viewmodel.get_image_paths()
            index = image_paths.index(unicodeFilePath)
            fileWidgetItem = self._file_list_widget.item(index)
            self._file_list_widget.setCurrentItem(fileWidgetItem)

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
        super(LabelerView, self).resizeEvent(event)

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
        self.viewmodel.close()
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
        images.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', x)])
        return images

    # ------------------------------------------------------------------
    # View command methods — called by actions / shortcuts
    # ------------------------------------------------------------------

    def _open_dir(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(self._current_image_path) if self._current_image_path else '.'
        if self.lastOpenDir is not None and len(self.lastOpenDir) > 1:
            path = self.lastOpenDir
        dirpath = self.tr(str(QFileDialog.getExistingDirectory(
            self,
            '%s - Open Directory' % __appname__,
            path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )))
        if not dirpath:
            return
        self.lastOpenDir = dirpath
        self._dir_name = dirpath
        self._current_image_path = None
        image_paths = self.scanAllImages(dirpath)
        self.viewmodel.save_labeler_image_list(image_paths)
        self.viewmodel.load_images(image_paths)

    def openPrevImg(self, _value=False):
        if not self.mayContinue():
            return
        self.viewmodel.navigate_to_previous(self._current_image_path)

    def openNextImg(self, _value=False):
        if not self.mayContinue():
            return
        self.viewmodel.navigate_to_next(self._current_image_path)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(self._current_image_path) if self._current_image_path else '.'
        formats = gui_util.SUPPORTED_IMAGE_FORMATS
        label_file_class = self._get_label_file_class()
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % label_file_class.SUFFIX])
        filename = QFileDialog.getOpenFileName(
            self, '%s - Choose Image or Label file' % __appname__, path, filters
        )
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self._load_file(filename)

    def _save_label(self, _value=False):
        if self._current_image_path is None or self._current_image.isNull():
            return
        self.viewmodel.save_label(
            self._current_image_path,
            self.canvas.get_shape(),
            self._current_image.width(),
            self._current_image.height(),
            self._current_image.isGrayscale(),
        )

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
        return QMessageBox.critical(self, title, '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self._current_image_path) if self._current_image_path else '.'

    def _delete_selected_shape(self):
        self._remove_shape(self.canvas.delete_selected())
        self._set_dirty()
        if self.no_shapes():
            for action in self.actions.on_shapes_present:
                action.setEnabled(False)

    def make_dataset_summary(self, dataset_summary_path, include_empty: bool):
        return build_dataset_summary(self.viewmodel.get_image_paths(), dataset_summary_path, include_empty)

    def _toggle_keep_prev(self):
        self._is_keep_prev = not self._is_keep_prev

    def _delete_label_file(self):
        selected_items = self._file_list_widget.selectedItems()
        target_paths = [self.tr(item.text()) for item in selected_items]
        if not target_paths and self._current_image_path is not None:
            target_paths = [self._current_image_path]
        if not target_paths:
            return
        mb = QMessageBox
        if len(target_paths) == 1:
            msg = self.tr(
                "You are about to permanently delete label file of {}, \r\nproceed anyway?"
            ).format(target_paths[0])
        else:
            msg = self.tr(
                "You are about to permanently delete {} label files, \r\nproceed anyway?"
            ).format(len(target_paths))
        if mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No) != mb.Yes:
            return
        self.viewmodel.delete_label_files(target_paths, self._current_image_path)

    def _remove_selected_images_from_list(self):
        selected_items = self._file_list_widget.selectedItems()
        target_paths = [self.tr(item.text()) for item in selected_items]
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
        if mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No) != mb.Yes:
            return
        self.viewmodel.remove_images_from_list(target_paths, self._current_image_path)

    def _select_all_files_in_list(self):
        if self._file_list_widget.count() == 0:
            return
        self._file_list_widget.setFocus(Qt.ShortcutFocusReason)
        self._file_list_widget.selectAll()
        self.status(f"Selected {self._file_list_widget.count()} images.", 2000)

    # ------------------------------------------------------------------
    # View slots — respond to ViewModel signals
    # ------------------------------------------------------------------

    def _on_image_index_started(self, total: int) -> None:
        if not self.isVisible():
            self._image_index_progress = None
            return
        self._image_index_progress = QProgressDialog(
            "Scanning image labels...", "Cancel", 0, total, self
        )
        self._image_index_progress.setWindowTitle("Indexing Images")
        self._image_index_progress.setWindowModality(Qt.WindowModal)
        self._image_index_progress.setMinimumDuration(0)
        self._image_index_progress.setValue(0)
        self._image_index_progress.canceled.connect(self.viewmodel.cancel_image_index)
        self._image_index_progress.show()

    def _on_image_index_progress(self, cur: int, total: int, path: str) -> None:
        if self._image_index_progress is None:
            return
        try:
            self._image_index_progress.setMaximum(max(total, 1))
            self._image_index_progress.setLabelText(
                f"Scanning image labels...\n{os.path.basename(path)}"
            )
            self._image_index_progress.setValue(cur)
        except RuntimeError:
            self._image_index_progress = None

    def _on_image_index_completed(self, was_cancelled: bool) -> None:
        if self._image_index_progress is not None:
            self._image_index_progress.close()
            self._image_index_progress.deleteLater()
            self._image_index_progress = None

    def _on_image_index_failed(self, message: str) -> None:
        if self._image_index_progress is not None:
            self._image_index_progress.close()
            self._image_index_progress.deleteLater()
            self._image_index_progress = None
        gui_util.get_message_box(self, "Indexing Failed", "Failed to scan image labels.")

    def _on_file_list_cleared(self) -> None:
        self._file_list_widget.clear()
        self._file_item_states.clear()

    def _on_file_list_batch_ready(self, batch: list) -> None:
        self._file_list_widget.setUpdatesEnabled(False)
        for path, has_label, is_empty, needs_confirm in batch:
            self._file_item_states[path] = (has_label, is_empty, needs_confirm)
            item = QListWidgetItem(path)
            item.setToolTip(path)
            self._apply_file_item_style(item, has_label, is_empty, needs_confirm)
            self._file_list_widget.addItem(item)
        self._file_list_widget.setUpdatesEnabled(True)

    def _on_file_list_item_state_changed(
        self, path: str, has_label: bool, is_empty: bool, needs_confirm: bool
    ) -> None:
        self._file_item_states[path] = (has_label, is_empty, needs_confirm)
        for item in self._file_list_widget.findItems(path, Qt.MatchExactly):
            self._apply_file_item_style(item, has_label, is_empty, needs_confirm)

    def _on_file_list_items_removed(self, removed_paths: set) -> None:
        for path in removed_paths:
            self._file_item_states.pop(path, None)
        for row in range(self._file_list_widget.count() - 1, -1, -1):
            item = self._file_list_widget.item(row)
            if self.tr(item.text()) in removed_paths:
                self._file_list_widget.takeItem(row)

    def _on_file_list_info_updated(self, total: int, labeled: int) -> None:
        self._lbl_file_list_info.setText(f"Total: {total}, Labeled: {labeled}")

    def _on_navigate_reset(self, next_path: str) -> None:
        self.resetState()
        self._set_clean()
        self.canvas.setEnabled(False)
        self._current_image_path = None
        self._current_image = QImage()
        if next_path:
            self._load_file(next_path)
        else:
            self._main_window.setWindowTitle(__appname__)

    def _on_label_files_deleted(self, deleted_paths: set, reload_path: str) -> None:
        if reload_path:
            self.resetState()
            self._load_file(reload_path, is_load_after_delete=True)

    @staticmethod
    def _apply_file_item_style(
        item: QListWidgetItem, has_label: bool, is_empty: bool, needs_confirm: bool
    ) -> None:
        """Apply color coding to a file list item based on label state."""
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

    def _auto_label(self):
        import torch
        from aligner_engine.detector import Detector

        def join_path(*paths):
            osp_path = os.path.join(*paths)
            return osp_path.replace('\\', '/')

        if not self.viewmodel.is_there_trained_checkpoint():
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

        img_paths = self.viewmodel.get_unlabeled_image_paths()

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
        image_paths = self.viewmodel.get_labeler_image_list()
        self.viewmodel.load_images(image_paths)

    def _item_selected_changed_file_list(self):
        indexes = self._file_list_widget.selectedIndexes()
        if not indexes:
            return
        item = self._file_list_widget.item(indexes[0].row())
        self._file_list_widget.scrollToItem(item)
        # When exactly one item is selected and we're not already inside _load_file
        # (handles arrow key navigation in addition to mouse clicks)
        if len(indexes) == 1 and not getattr(self, '_loading_file', False):
            target_path = self.tr(item.text())
            if target_path != self._current_image_path:
                self._load_file(target_path)


class _FileListWidget(QListWidget):
    """QListWidget that navigates with plain arrow keys without extending the selection."""

    _NAV_KEYS = {Qt.Key_Up, Qt.Key_Down, Qt.Key_Home, Qt.Key_End, Qt.Key_PageUp, Qt.Key_PageDown}

    def keyPressEvent(self, event):
        if event.key() in self._NAV_KEYS and event.modifiers() == Qt.NoModifier:
            current = self.currentRow()
            count = self.count()
            if count == 0:
                return
            key = event.key()
            if key == Qt.Key_Up:
                new_row = max(0, current - 1)
            elif key == Qt.Key_Down:
                new_row = min(count - 1, current + 1)
            elif key == Qt.Key_Home:
                new_row = 0
            elif key == Qt.Key_End:
                new_row = count - 1
            elif key == Qt.Key_PageUp:
                visible = max(1, self.height() // max(1, self.sizeHintForRow(0)))
                new_row = max(0, current - visible)
            else:  # PageDown
                visible = max(1, self.height() // max(1, self.sizeHintForRow(0)))
                new_row = min(count - 1, current + visible)
            self.setCurrentRow(new_row)
        else:
            super().keyPressEvent(event)


class ElideLeftDelegate(QStyledItemDelegate):
    def __init__(self):
        super().__init__()

    def paint(self, painter, option, index):
        painter.save()
        value = index.data(Qt.DisplayRole)
        textcolor = index.model().data(index, Qt.TextColorRole)
        pen = QPen(option.palette.text().color() if textcolor is None else textcolor.color())
        painter.setPen(pen)
        if int(option.state & QStyle.State_Selected) > 0:
            painter.fillRect(option.rect, QColor(20, 100, 160))
        else:
            backgroundColor = index.model().data(index, Qt.BackgroundColorRole)
            if backgroundColor:
                painter.fillRect(option.rect, backgroundColor.color())
        painter.drawText(option.rect, Qt.AlignLeft | Qt.AlignVCenter,
                         option.fontMetrics.elidedText(str(value) if value else '',
                                                       Qt.ElideLeft,
                                                       option.rect.width()))
        painter.restore()


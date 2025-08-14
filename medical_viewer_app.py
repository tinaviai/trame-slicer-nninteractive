# -*- coding:utf-8 -*-
import pathlib
import numpy as np
import tempfile

from trame.decorators import TrameApp, change
from trame.app import get_server
from trame_server import Server
from trame.widgets import vuetify3
from trame.widgets import html

from trame_slicer.rca_view import register_rca_factories
from trame_vuetify.ui.vuetify3 import SinglePageWithDrawerLayout
from trame_client.widgets.html import Div

from trame_slicer.core.markups_logic import MarkupsLogic
from trame_slicer.core import SlicerApp, LayoutManager, SegmentationEditor
from trame_slicer.segmentation import SegmentationEffectID, SegmentationPaintEffect, BrushShape, SegmentPaintWidget2D

from slicer import (
    vtkMRMLTransformNode,
    vtkMRMLSegmentationNode,
    vtkMRMLMarkupsNode,
    vtkMRMLInteractionNode,
    vtkSlicerSegmentationsModuleLogic,
    vtkSlicerVolumesLogic,
)
import slicer.util

from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonTransforms import vtkGeneralTransform
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonCore import vtkStringArray, vtkCommand

from widgets.utils import StateId
from widgets.load_client_volume_files_button import LoadClientVolumeFilesButton
from widgets.segment_editor import SegmentEditor, arrayFromSegmentBinaryLabelmap

from utils import PromptManager

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent.parent


@TrameApp()
class NninteractiveTrameSlicerApp:
    def __init__(self, server=None):
        self._server = get_server(server, client_type="vue3")
        self._slicer_app = SlicerApp()

        register_rca_factories(self._slicer_app.view_manager, self._server)

        self._layout_manager = LayoutManager(
            self._slicer_app.scene,
            self._slicer_app.view_manager,
            self._server.ui.layout_grid,
        )

        self._layout_manager.register_layout_dict(
            LayoutManager.default_grid_configuration()
        )

        self.server.state.setdefault(
            StateId.vr_preset_value,
            "CT-Coronary-Arteries-3",
        )

        default_layout = "Quad View"  # "Quad View" # "Axial Primary"
        self.server.state.setdefault(StateId.current_layout_name, default_layout)

        # Update the layout to the default layout
        self._layout_manager.set_layout(default_layout)

        self.segment_editor = None
        self._build_ui()

        self.state["pos_neg_state"] = 0  # 0 : pos, 1: neg
        # self.state["unselected_style"] = "border-radius: 0px; border: 0px solid #424949;"
        self.state["selected_style"] = "border-radius: 5px; border: 3px solid grey;"
        self.state["show_volume_rendering"] = True
        self.state["can_lasso_clear"] = False
        self.previous_states = {}

        self._volume_node = None
        self.point_pts_node = None
        self.bbox_roi_node = None
        self.lasso_curve_node = None
        self.scribble_segment_node_name = "ScribbleSegmentNode (do not touch)"

        self._slicer_app.app_logic.GetInteractionNode().AddObserver(vtkMRMLInteractionNode.InteractionModeChangedEvent,
                                                                    self.on_interaction_node_modified)

        self.prompt_manager = PromptManager()

        @self._server.controller.trigger("download_binary")
        def download():
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = pathlib.Path(tmp_dir)
                self._slicer_app.io_manager.write_segmentation(
                    self.segment_editor._segmentation_node,
                    tmp_dir / "tmp.nrrd")
                return self._server.protocol.addAttachment((tmp_dir / "tmp.nrrd").read_bytes())

    @property
    def state(self):
        return self.server.state

    @property
    def server(self) -> Server:
        return self._server

    @property
    def markups_logic(self) -> MarkupsLogic:
        return self._slicer_app.markups_logic

    def _build_ui(self):
        with SinglePageWithDrawerLayout(self._server) as self.ui:
            self.ui.root.theme = "dark"
            self.ui.title.set_text("NNInterActive")
            with self.ui.toolbar:
                vuetify3.VSpacer()
                seg_name = "seg.nrrd"
                html.Button(
                    "Download Segmentation",
                    # v_if="pos_neg_state===1?true:false",
                    click=f"utils.download('{seg_name}', trigger('download_binary'), 'application/octet-stream')",
                    style="margin-right: 20px"
                )
            with (self.ui.content, Div(classes="fill-height d-flex flex-row flex-grow-1")):
                self._server.ui.layout_grid(self.ui)
            with self.ui.drawer as drawer:
                LoadClientVolumeFilesButton(self.server, self._slicer_app, self.set_image2prompt_manager)
                with vuetify3.VRow(no_gutters=True, style=f"margin-left: 0px;"):
                    vuetify3.VCheckbox(
                        label="Show Volume Rendering",
                        v_model=("show_volume_rendering",),
                        hide_details=True,
                        click="$event.stopPropagation();",
                        update_modelValue=f"""
                                          trigger('{self.server.trigger_name(self.on_toggle_volume_visibility)}', [$event]);
                                          """,
                    )
                # positive and negative button
                with vuetify3.VRow(no_gutters=True):
                    with vuetify3.VCol():
                        vuetify3.VBtn("positive", block=True,
                                      style=("pos_neg_state===0?selected_style:unselected_style",),
                                      click=lambda: self.select_pos_neg_state(0))
                    with vuetify3.VCol():
                        vuetify3.VBtn("negative", block=True,
                                      style=("pos_neg_state===1?selected_style:unselected_style",),
                                      click=lambda: self.select_pos_neg_state(1))
                # interactive button
                # point
                with vuetify3.VRow(no_gutters=True):
                    with vuetify3.VCol():
                        vuetify3.VBtn("Point", block=True, click=lambda: self.add_point_interaction())
                    # with vuetify3.VCol():
                    #     vuetify3.VBtn("Update", block=True, click=lambda: self.update_point_interaction())
                # bbox
                with vuetify3.VRow(no_gutters=True):
                    with vuetify3.VCol():
                        vuetify3.VBtn("Bounding Box", block=True, click=lambda: self.add_bbox_interaction())
                    # with vuetify3.VCol():
                    #     vuetify3.VBtn("Update", block=True, click=lambda: self.update_bbox_interaction())
                # scribble
                with vuetify3.VRow(no_gutters=True):
                    with vuetify3.VCol():
                        vuetify3.VBtn("Scribble", block=True, click=lambda: self.add_scribble_interaction())
                # lasso
                with vuetify3.VRow(no_gutters=True):
                    with vuetify3.VCol():
                        vuetify3.VBtn("Lasso", block=True, click=lambda: self.add_lasso_interaction())
                    with vuetify3.VCol(v_show="can_lasso_clear"):
                        vuetify3.VBtn("Clear", block=True, click=lambda: self.on_lasso_cancel_clicked())

                self.segment_editor = SegmentEditor(self.server, self._slicer_app)

    def set_image2prompt_manager(self, volume_node):
        nshape = tuple(reversed(volume_node.GetImageData().GetDimensions()))
        vimage = volume_node.GetImageData()
        narray = numpy_support.vtk_to_numpy(vimage.GetPointData().GetScalars()).reshape(nshape)

        self._volume_node = volume_node
        self.prompt_manager.set_image(narray)

    def update_segment2prompt_manager(self):
        segment_data = self.segment_editor.get_current_segment_data()
        selected_segment_changed = self.selected_segment_changed(segment_data)
        if selected_segment_changed:
            segment_mask = segment_data.astype(np.uint8)
            self.prompt_manager.set_segment(segment_mask)

    def on_toggle_volume_visibility(self, visibility):
        if self._volume_node is not None:
            volume_rendering = self._slicer_app.volume_rendering
            display_node = volume_rendering.get_vr_display_node(self._volume_node)
            display_node.SetVisibility(visibility)

    def select_pos_neg_state(self, pos_neg_state):
        self.state["pos_neg_state"] = pos_neg_state

    @property
    def is_positive(self):
        return self.state["pos_neg_state"] == 0

    def add_point_interaction(self):
        node_class = "vtkMRMLMarkupsFiducialNode"
        node_name = "Point_" + ("positive" if self.is_positive else "negative")
        node = self.point_pts_node
        if node is not None:
            self.remove_node_by_names([node.name])
        node = self._slicer_app.scene.AddNewNodeByClass(node_class, node_name)
        node.CreateDefaultDisplayNodes()

        pt_color = [0, 0, 1] if self.is_positive else [1, 0, 0]
        display_node = node.GetDisplayNode()
        # display_node.SetSelectedColor(*pt_color)
        # display_node.SetActiveColor(*pt_color)
        self.display_node_markup_point(display_node, pt_color)

        node.AddObserver(vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.update_point_interaction)
        self.markups_logic.place_node(node, persistent=False)

        self.point_pts_node = node

    def add_point_interaction2(self):
        node_class = "vtkMRMLMarkupsFiducialNode"
        node_name = "Point_" + ("positive" if self.is_positive else "negative")
        node = self.point_pts_node
        if node is not None:
            self.remove_node_by_names([node.name])
        node = self._slicer_app.scene.AddNewNodeByClass(node_class, node_name)
        node.CreateDefaultDisplayNodes()
        node_id = node.GetID()

        pt_color = [0, 0, 1] if self.is_positive else [1, 0, 0]
        display_node = node.GetDisplayNode()
        # display_node.SetSelectedColor(*pt_color)
        # display_node.SetActiveColor(*pt_color)
        self.display_node_markup_point(display_node, pt_color)

        node.AddObserver(vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.update_point_interaction)
        self.on_place_button_clicked(True, "point", node_class, node_id)

        self.point_pts_node = node

    def update_point_interaction(self, caller=None, event=None):
        node = caller
        if node is None:
            # node_name = "Point_" + ("positive" if self.is_positive else "negative")
            node = self.point_pts_node
        if node is None:
            return
        n = node.GetNumberOfControlPoints()
        pos = node.GetNthControlPointPositionWorld(n - 1)

        xyz = self.ras_to_xyz(pos)

        volume_node = self._volume_node
        if volume_node:
            point_coordinates = xyz[::-1]
            # 同步分割
            self.update_segment2prompt_manager()
            segment_mask = self.prompt_manager.add_point_interaction(point_coordinates, self.is_positive)

            # labelmap_node = self._slicer_app.scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TempLabel")

            # segmentation_name = "MySegmentation"
            # segmentation_node = self._slicer_app.scene.GetFirstNodeByName(segmentation_name)
            # if not segmentation_node:
            #     segmentation_node = self._slicer_app.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", segmentation_name)
            #     segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self._volume_node)
            #
            # referenceVolumeNode = segmentation_node.GetNodeReference(vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole())
            # labelmap_node = self._slicer_app.volumes_logic.CreateAndAddLabelVolume(referenceVolumeNode, "__temp__")
            # slicer.util.updateVolumeFromArray(labelmap_node, segment_mask)
            #
            # segment_name = "Segment_1"
            # segment_id = segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
            # if not segment_id:
            #     segment_id = segmentation_node.GetSegmentation().AddEmptySegment(
            #         segment_name
            #     )
            self.show_segment(segment_mask)

    def add_bbox_interaction(self):
        node_class = "vtkMRMLMarkupsROINode"
        node_name = "BBox_" + ("positive" if self.is_positive else "negative")
        node = self.bbox_roi_node
        if node is not None:
            self.remove_node_by_names([node.name])
        node = self._slicer_app.scene.AddNewNodeByClass(node_class, node_name)
        node.CreateDefaultDisplayNodes()
        node_id = node.GetID()

        pt_color = [0, 0, 1] if self.is_positive else [1, 0, 0]
        display_node = node.GetDisplayNode()
        # display_node.SetSelectedColor(*pt_color)
        # display_node.SetActiveColor(*pt_color)
        self.display_node_markup_bbox(display_node, pt_color)

        self.prev_caller = None
        node.AddObserver(vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.update_bbox_interaction)
        self.on_place_button_clicked(True, "bbox", node_class, node_id)
        # self.markups_logic.place_node(node, persistent=False)

        self.bbox_roi_node = node

    def update_bbox_interaction(self, caller=None, event=None):
        node = caller
        if node is None:
            # node_name = "BBox_" + ("positive" if self.is_positive else "negative")
            node = self.bbox_roi_node
        if node is None:
            return
        n = node.GetNumberOfControlPoints()
        pos = node.GetNthControlPointPositionWorld(n - 1)

        xyz = self.ras_to_xyz(pos)
        if self.prev_caller is not None and node.GetID() == self.prev_caller.GetID():
            roi_node = self._slicer_app.scene.GetNodeByID(self.prev_caller.GetID())
            current_size = list(roi_node.GetSize())
            drawn_in_axis = np.argwhere(np.array(xyz) == self.prev_bbox_xyz).squeeze()
            current_size[drawn_in_axis] = 0
            roi_node.SetSize(current_size)

            volume_node = self._volume_node
            if volume_node:
                outer_point_two = self.prev_bbox_xyz

                outer_point_one = [
                    xyz[0] * 2 - outer_point_two[0],
                    xyz[1] * 2 - outer_point_two[1],
                    xyz[2] * 2 - outer_point_two[2],
                ]

                # 同步分割
                self.update_segment2prompt_manager()
                segment_mask = self.prompt_manager.add_bbox_interaction(outer_point_one[::-1], outer_point_two[::-1],
                                                                        self.is_positive)
                self.show_segment(segment_mask)
                # 清空当前输入
                self.remove_node_by_names([self.bbox_roi_node.name])
                self.bbox_roi_node = None
                # 取消绘制状态
                interaction_node = self._slicer_app.app_logic.GetInteractionNode()
                interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

            self.prev_caller = None
        else:
            self.prev_bbox_xyz = xyz

        self.prev_caller = node

    def add_scribble_interaction(self):
        if self._slicer_app.scene.GetFirstNodeByName(self.scribble_segment_node_name) is None:
            self.setup_scribble_prompt()
        interaction_node = self._slicer_app.app_logic.GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

        segment_id = "fg" if self.is_positive else "bg"
        self.scribble_editor_node.set_active_segmentation(self.scribble_segmentation_node, self._volume_node)
        self.scribble_editor_node.set_active_segment_id(segment_id)

        # Activate paint effect
        paint_effect: SegmentationPaintEffect = self.scribble_editor_node.set_active_effect_id(
            SegmentationEffectID.Paint)
        if paint_effect:
            # paint_effect._brush_model.set_sphere_parameters(8.0, 16, 16)
            # paint_effect._brush_model.set_shape(BrushShape.Sphere)
            # 上面的方法无效，底层代码写死了使用Cylinder, segment_paint_widget_2d.py Line:168
            # 暂用下面的方式修改笔刷直径
            for widget in paint_effect.get_widgets():
                paintWidget: SegmentPaintWidget2D = widget
                # vSize = paintWidget.view.render_window().GetScreenSize()[1]
                # relative_brush_size = 5
                paintWidget._brush_diameter_pix = 16  # (relative_brush_size / 100) * vSize

            # paint_effect.setParameter("BrushUseAbsoluteSize", "0")  # Use relative mode
            # paint_effect.setParameter("BrushSphere", "0")  # 2D brush
            # paint_effect.setParameter("BrushRelativeDiameter", ".75")
            self._scribble_labelmap_callback_tag = {
                "tag": self.scribble_segmentation_node.AddObserver(
                    vtkCommand.AnyEvent, self.on_scribble_finished
                ),
                "label_name": segment_id,
            }
        pass

    def setup_scribble_prompt(self):
        self.scribble_editor_node = SegmentationEditor(self._slicer_app.scene, self._slicer_app.segmentation_logic,
                                                       self._slicer_app.view_manager)
        self.scribble_segmentation_node = self._slicer_app.scene.AddNewNodeByClass("vtkMRMLSegmentationNode",
                                                                                   self.scribble_segment_node_name)
        self.scribble_segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self._volume_node)
        self.scribble_editor_node.set_active_segmentation(self.scribble_segmentation_node, self._volume_node)

        self.scribble_segmentation_node.CreateDefaultDisplayNodes()
        self.scribble_segmentation_node.GetSegmentation().AddEmptySegment(
            "bg", "bg", [1.0, 0.0, 0.0]
        )
        self.scribble_segmentation_node.GetSegmentation().AddEmptySegment(
            "fg", "fg", [0.0, 0.0, 1.0]
        )
        dn = self.scribble_segmentation_node.GetDisplayNode()

        opacity = 0.2
        dn.SetSegmentOpacity2DFill("bg", opacity)
        dn.SetSegmentOpacity2DOutline("bg", opacity)
        dn.SetSegmentOpacity2DFill("fg", opacity)
        dn.SetSegmentOpacity2DOutline("fg", opacity)

        self._prev_scribble_masks = {"bg": None, "fg": None}
        pass

    def on_scribble_finished(self, caller, event):
        # Clean up observer if you only want it once
        if hasattr(self, "_scribble_labelmap_callback_tag"):
            caller.RemoveObserver(self._scribble_labelmap_callback_tag["tag"])
            label_name = self._scribble_labelmap_callback_tag["label_name"]
            del self._scribble_labelmap_callback_tag
        else:
            return

        mask = arrayFromSegmentBinaryLabelmap(self._slicer_app,
                                              self.scribble_segmentation_node, label_name, self._volume_node
                                              )

        if (
                hasattr(self, "_prev_scribble_masks")
                and self._prev_scribble_masks[label_name] is not None
        ):
            prev_scribble_mask = self._prev_scribble_masks[label_name]
        else:
            prev_scribble_mask = mask * 0

        diff_mask = mask - prev_scribble_mask
        self._prev_scribble_masks[label_name] = mask

        # 同步分割
        self.update_segment2prompt_manager()
        segment_mask = self.prompt_manager.add_scribble_interaction(diff_mask, include_interaction=self.is_positive)

        self.show_segment(segment_mask)

        # Deactivate paint effect
        if self.scribble_editor_node:
            self.scribble_editor_node.set_active_effect(None)  # Clears the active effect

        # Optionally clear or reset the segmentation node
        if hasattr(self, "_scribble_labelmap_callback_tag"):
            tag = self._scribble_labelmap_callback_tag.get("tag", None)
            if tag:
                self.scribble_segmentation_node.RemoveObserver(tag)
            del self._scribble_labelmap_callback_tag

        return
        # self.ui.pbInteractionScribble.click()  # turn it off
        # self.ui.pbInteractionScribble.click()  # turn it on
        pass

    def add_lasso_interaction(self):
        node_class = "vtkMRMLMarkupsClosedCurveNode"
        node_name = "Lasso_" + ("positive" if self.is_positive else "negative")
        node = self.lasso_curve_node
        if node is not None:
            self.remove_node_by_names([node.name])
        node = self._slicer_app.scene.AddNewNodeByClass(node_class, node_name)
        node.SetCurveTypeToLinear()
        node.CreateDefaultDisplayNodes()
        node_id = node.GetID()

        pt_color = [0, 0, 1] if self.is_positive else [1, 0, 0]
        display_node = node.GetDisplayNode()
        # display_node.SetSelectedColor(*pt_color)
        # display_node.SetActiveColor(*pt_color)
        self.display_node_markup_lasso(display_node, pt_color)

        self.prev_caller = None
        node.AddObserver(vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.update_lasso_interaction)
        self.on_place_button_clicked(True, "lasso", node_class, node_id)
        # self.markups_logic.place_node(node, persistent=False)

        self.lasso_curve_node = node
        pass

    def update_lasso_interaction(self, caller=None, event=None):
        if self.lasso_curve_node is not None:
            pointsDefined = self.lasso_curve_node.GetNumberOfControlPoints() > 0
            self.state["can_lasso_clear"] = pointsDefined
            self.state.flush()

    def on_lasso_cancel_clicked(self):
        if self.lasso_curve_node is not None:
            self.lasso_curve_node.RemoveAllControlPoints()
        self.state["can_lasso_clear"] = False
        self.state.flush()

    def submit_lasso_if_present(self):
        # node_name = "ROI_" + ("positive" if self.is_positive else "negative")
        node = self.lasso_curve_node
        if node is None:
            return

        xyzs = self.xyz_from_caller(node, point_type="curve_point")

        if len(xyzs) < 3:
            return

        mask = self.lasso_points_to_mask(xyzs)

        volume_node = self._volume_node
        if volume_node:
            # 同步分割
            self.update_segment2prompt_manager()
            segment_mask = self.prompt_manager.add_lasso_interaction(mask, include_interaction=self.is_positive)

            self.show_segment(segment_mask)
            # 清空当前输入
            self.remove_node_by_names([self.lasso_curve_node.name])
            self.lasso_curve_node = None
            self.state["can_lasso_clear"] = False
            self.state.flush()
            # 取消绘制状态
            interaction_node = self._slicer_app.app_logic.GetInteractionNode()
            interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

    def lasso_points_to_mask(self, points):
        """
        Given a list of voxel coords (defining a polygon in one slice),
        returns a 3D mask with that polygon filled in the appropriate slice.
        """
        from skimage.draw import polygon

        shape = slicer.util.arrayFromVolume(self._volume_node).shape
        pts = np.array(points)  # shape (n, 3)

        # Determine which coordinate is constant
        const_axes = [i for i in range(3) if np.unique(pts[:, i]).size == 1]
        if len(const_axes) != 1:
            raise ValueError(
                "Expected exactly one constant coordinate among the points"
            )
        const_axis = const_axes[0]
        const_val = int(pts[0, const_axis])

        # Create a blank 3D mask
        mask = np.zeros(shape, dtype=np.uint8)

        # Depending on which axis is constant, extract the 2D polygon and fill the corresponding slice.
        # Note: our volume is ordered as (z, y, x)
        if const_axis == 2:
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            rr, cc = polygon(y_coords, x_coords, shape=(shape[1], shape[2]))
            mask[const_val, rr, cc] = 1
        elif const_axis == 1:
            x_coords = pts[:, 0]
            z_coords = pts[:, 2]
            rr, cc = polygon(z_coords, x_coords, shape=(shape[0], shape[2]))
            mask[rr, const_val, cc] = 1
        elif const_axis == 0:
            y_coords = pts[:, 1]
            z_coords = pts[:, 2]
            rr, cc = polygon(z_coords, y_coords, shape=(shape[0], shape[1]))
            mask[rr, cc, const_val] = 1

        return mask

    # prompt_name: point, bbox, lasso
    def on_place_button_clicked(self, checked, prompt_name, node_class, node_id, persistent=False):
        interactionNode = self._slicer_app.app_logic.GetInteractionNode()
        if checked:
            selectionNode = self._slicer_app.app_logic.GetSelectionNode()
            selectionNode.SetReferenceActivePlaceNodeClassName(node_class)
            selectionNode.SetActivePlaceNodeID(node_id)
            interactionNode.SetPlaceModePersistence(persistent)
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        else:
            if prompt_name == "lasso":
                self.submit_lasso_if_present()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def on_interaction_node_modified(self, caller, event):
        """
        Deselect prompt button if interaction mode is not place point anymore
        """
        interactionNode = self._slicer_app.app_logic.GetInteractionNode()
        selectionNode = self._slicer_app.app_logic.GetSelectionNode()
        markups = [self.point_pts_node, self.bbox_roi_node, self.lasso_curve_node]
        for node in markups:
            if node:
                if interactionNode.GetCurrentInteractionMode() != vtkMRMLInteractionNode.Place:
                    if "Lasso" in node.name:  # and (self.ui.pbInteractionLasso.isChecked()):
                        self.submit_lasso_if_present()
                    # prompt_type["button"].setChecked(False)
                elif interactionNode.GetCurrentInteractionMode() == vtkMRMLInteractionNode.Place:
                    placingThisNode = (selectionNode.GetActivePlaceNodeID() == node.GetID())
                    # prompt_type["button"].setChecked(placingThisNode)

        # Stop scribble if placing markup
        if interactionNode.GetCurrentInteractionMode() == vtkMRMLInteractionNode.Place:
            # self.ui.pbInteractionScribble.setChecked(False)
            pass

    def display_node_markup_point(self, display_node, pt_color):
        """
        Handles the appearance of the point display node.
        """
        # display_node.SetTextScale(0)  # Hide text labels
        display_node.SetGlyphScale(0.75)  # Make the points larger
        display_node.SetColor(*pt_color)  # Green color
        display_node.SetSelectedColor(*pt_color)
        display_node.SetActiveColor(*pt_color)
        display_node.SetOpacity(1.0)  # Fully opaque
        # display_node.SetSliceProjection(False)  # Make points visible in all slice views

    def display_node_markup_bbox(self, display_node, pt_color):
        """
        Handles the appearance of the BBox display node.
        """
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(0.5)
        display_node.SetSelectedColor(*pt_color)
        display_node.SetColor(*pt_color)
        display_node.SetActiveColor(*pt_color)
        display_node.SetSliceProjectionColor(*pt_color)
        display_node.SetInteractionHandleScale(1)
        display_node.SetGlyphScale(0)
        display_node.SetHandlesInteractive(False)
        # display_node.SetTextScale(0)

    def display_node_markup_lasso(self, display_node, pt_color):
        """
        Handles the appearance of the lasso display node.
        """
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(0.5)
        display_node.SetSelectedColor(*pt_color)
        display_node.SetColor(*pt_color)
        display_node.SetActiveColor(*pt_color)
        display_node.SetSliceProjectionColor(*pt_color)
        display_node.SetGlyphScale(1)
        display_node.SetLineThickness(0.3)
        display_node.SetHandlesInteractive(False)
        # display_node.SetTextScale(0)

    def show_segment(self, segment_mask):
        self.previous_states["segment_data"] = segment_mask

        segmentation_node, segment_id = (
            self.segment_editor.get_selected_segmentation_node_and_segment_id()
        )
        referenceVolumeNode = segmentation_node.GetNodeReference(
            vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole())
        labelmap_node = self._slicer_app.volumes_logic.CreateAndAddLabelVolume(referenceVolumeNode, "__temp__")
        slicer.util.updateVolumeFromArray(labelmap_node, segment_mask)

        segmentIds = vtkStringArray()
        segmentIds.InsertNextValue(segment_id)
        self._slicer_app.segmentation_logic.ImportLabelmapToSegmentationNode(labelmap_node, segmentation_node,
                                                                             segmentIds)
        segmentation_node.Modified()
        # segmentation_node.CreateClosedSurfaceRepresentation()

        del segment_mask
        # display_node = segmentation_node.GetDisplayNode()
        # display_node.SetVisibility(True)
        self._slicer_app.scene.RemoveNode(labelmap_node)

    def selected_segment_changed(self, segment_data=None):
        """
        Checks if the current segment mask has changed from our `self.previous_states`.
        """
        if segment_data is None:
            segment_data = self.segment_editor.get_current_segment_data()
        old_segment_data = self.previous_states.get("segment_data", None)
        selected_segment_changed = old_segment_data is None or not np.array_equal(
            old_segment_data.astype(bool), segment_data.astype(bool)
        )

        if old_segment_data is not None:
            print(f"old_segment_data.sum(): {old_segment_data.sum()}")
        else:
            print("old_segment_data is None")

        print(f"selected_segment_changed: {selected_segment_changed}")

        return selected_segment_changed

    def ras_to_xyz(self, pos):
        """
        Converts an RAS position to IJK voxel coords in the current volume node.
        """
        volume_node = self._volume_node

        transform_ras2volume_ras = vtkGeneralTransform()
        vtkMRMLTransformNode.GetTransformBetweenNodes(None, volume_node.GetParentTransformNode(),
                                                      transform_ras2volume_ras)
        point_Volume_ras = transform_ras2volume_ras.TransformPoint(pos)
        volume_ras2ijk = vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(volume_ras2ijk)
        point_ijk = [0, 0, 0, 1]
        volume_ras2ijk.MultiplyPoint(list(point_Volume_ras) + [1.0], point_ijk)
        xyz = [int(round(c)) for c in point_ijk[0:3]]
        return xyz

    def xyz_from_caller(self, caller, point_type="control_point"):
        """
        Extract voxel coordinates from a Markups node.
        `point_type` can be either "control_point" or "curve_point".
        """
        if point_type == "control_point":
            n = caller.GetNumberOfControlPoints()
            if n < 0:
                # debug_print("No control points found")
                return

            pos = [0, 0, 0]
            caller.GetNthControlPointPosition(n - 1, pos)
            # if lock_point:
            #     caller.SetNthControlPointLocked(n - 1, True)
            xyz = self.ras_to_xyz(pos)
            return xyz
        elif point_type == "curve_point":
            vtk_pts = caller.GetCurvePointsWorld()

            if vtk_pts is not None:
                vtk_pts_data = numpy_support.vtk_to_numpy(vtk_pts.GetData())
                xyz = [self.ras_to_xyz(pos) for pos in vtk_pts_data]
                # debug_print(xyz)
                return xyz

            return []
        else:
            raise ValueError(f'Unknown point_type {point_type}')

    def remove_node_by_names(self, node_names):
        for name in node_names:
            node = self._slicer_app.scene.GetFirstNodeByName(name)
            if node:
                self._slicer_app.scene.RemoveNode(node)


def main(server=None, **kwargs):
    app = NninteractiveTrameSlicerApp(server=server)
    app.server.start(**kwargs)


if __name__ == "__main__":
    main(host="0.0.0.0")

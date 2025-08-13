from __future__ import annotations

# from networkx.algorithms.bipartite import color
from trame.decorators import TrameApp, change
from trame.widgets.vuetify3 import (
    Template,
    VCard,
    VCardText,
    VCheckbox,
    VColorPicker,
    VIcon,
    VListItem,
    VMenu,
    VRow,
    VCol,
    VSlider,
    VTextField,
    VTable,
    VContainer,
    VBtn,
)
from trame_client.widgets.html import Span, Div, Thead, Th, Tbody, Tr, Td
from trame_vuetify.widgets.vuetify3 import VSelect, VSpacer
from undo_stack import Signal, UndoStack

from trame_slicer.core import SegmentationEditor, SlicerApp
from trame_slicer.segmentation import (
    SegmentationEffectID,
    SegmentationEraseEffect,
    SegmentationOpacityEnum,
    SegmentationPaintEffect,
    SegmentationScissorEffect,
    SegmentProperties,
    SegmentPaintWidget2D,
)
from trame_slicer.utils import (
    connect_all_signals_emitting_values_to_state,
)

from .control_button import ControlButton
from .utils import IdName, StateId, get_current_volume_node


class SegmentationId:
    current_segment_name = IdName()
    current_segment_id = IdName()
    is_renaming_segment = IdName()
    segments = IdName()
    segment_opacity_mode = IdName()
    opacity_2d = IdName()
    opacity_3d = IdName()


class SegmentationRename(Template):
    validate_clicked = Signal(str, str)
    cancel_clicked = Signal()

    segment_name_id = IdName()
    segment_color_id = IdName()

    def __init__(self, server, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._server = server

        with self:
            with VContainer(classes="pa-1 fill-height"):
                with VRow(style="width: 100%", align="center", align_content="center"):
                    VTextField(
                        density="compact",
                        v_model=(self.segment_name_id,),
                        hide_details="auto",
                        width="155"
                    )
                    ControlButton(
                        name="Validate new name",
                        icon="mdi-check",
                        click=self.on_validate_modify,
                        size=20,
                        density="compact",
                    )
                    ControlButton(
                        name="Cancel",
                        icon="mdi-close",
                        click=self.cancel_clicked,
                        size=20,
                        density="compact",
                    )

                with VRow(style="width: 100%", align="center"):
                    VColorPicker(
                        style="width: auto",
                        tile=True,
                        v_model=(self.segment_color_id,),
                        modes=("['rgb']",),
                    )

    def on_validate_modify(self):
        self.validate_clicked(
            self.state[self.segment_name_id],
            self.state[self.segment_color_id],
        )

    def set_segment_name(self, segment_name):
        self.state[self.segment_name_id] = segment_name

    def set_segment_color(self, color_hex: str):
        self.state[self.segment_color_id] = color_hex


class SegmentationOpacityModeToggleButton(ControlButton):
    def __init__(self, *args, **kwargs) -> None:
        value_to_icon = {
            SegmentationOpacityEnum.FILL.value: "mdi-circle-medium",
            SegmentationOpacityEnum.OUTLINE.value: "mdi-circle-outline",
            SegmentationOpacityEnum.BOTH.value: "mdi-circle",
        }

        super().__init__(
            *args,
            **kwargs,
            icon=(f"{{{{ {value_to_icon}[{SegmentationId.segment_opacity_mode}] }}}}",),
        )

        with self:
            pass


class SegmentSelection(Template):
    add_segment_clicked = Signal()
    delete_current_segment_clicked = Signal()
    start_rename_clicked = Signal()
    no_tool_clicked = Signal()
    paint_clicked = Signal()
    erase_clicked = Signal()
    scissors_clicked = Signal()
    toggle_3d_clicked = Signal()
    segment_visibility_toggled = Signal(str, bool)
    select_segment_id_clicked = Signal(str)
    opacity_mode_clicked = Signal()
    opacity_2d_changed = Signal()
    opacity_3d_changed = Signal()
    undo_clicked = Signal()
    redo_clicked = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self:
            # with (
            #     VRow(align="center"),
            #     VSelect(
            #         label="Current Segment",
            #         v_model=(SegmentationEditor.active_segment_id_changed.name,),
            #         items=(SegmentationId.segments,),
            #         item_value="props.segment_id",
            #         item_title="title",
            #         no_data_text="",
            #         hide_details="auto",
            #         min_width=200,
            #     ),
            # ):
            #     with (
            #         Template(v_slot_item="{props}"),
            #         VListItem(v_bind="props", color=""),
            #         Template(v_slot_prepend=""),
            #     ):
            #         VCheckbox(
            #             v_model=("props.visibility",),
            #             color=("props.color_hex",),
            #             base_color=("props.color_hex",),
            #             hide_details=True,
            #             click="$event.stopPropagation();",
            #             update_modelValue=f"""
            #                 trigger('{self.server.trigger_name(self.segment_visibility_toggled)}', [props.segment_id, $event]);
            #             """,
            #         )
            #     with Template(v_slot_selection="{item}"):
            #         VIcon("mdi-square", color=("item.props.color_hex",))
            #         Span("{{item.title}}", classes="pl-2")

            with VContainer(classes="pa-1 fill-height"):
                with VRow():
                    ControlButton(
                        name="Add new segment",
                        icon="mdi-plus-circle",
                        size=0,
                        click=self.add_segment_clicked,
                    )
                    ControlButton(
                        name="Delete current segment",
                        icon="mdi-minus-circle",
                        size=0,
                        click=self.delete_current_segment_clicked,
                    )
                    ControlButton(
                        name="Rename current segment",
                        icon="mdi-rename-box-outline",
                        size=0,
                        click=self.start_rename_clicked,
                    )
                    ControlButton(
                        name="Toggle 3D",
                        icon="mdi-video-3d",
                        size=0,
                        click=self.toggle_3d_clicked,
                        active=(f"{SegmentationEditor.show_3d_changed.name}",),
                    )

                with VRow():
                    ControlButton(
                        name="No tool",
                        icon="mdi-cursor-default",
                        size=0,
                        click=self.no_tool_clicked,
                        active=self.button_active(None),
                    )
                    ControlButton(
                        name="Paint",
                        icon="mdi-brush",
                        size=0,
                        click=self.paint_clicked,
                        active=self.button_active(SegmentationPaintEffect),
                    )
                    ControlButton(
                        name="Erase",
                        icon="mdi-eraser",
                        size=0,
                        click=self.erase_clicked,
                        active=self.button_active(SegmentationEraseEffect),
                    )
                    ControlButton(
                        name="Scissors",
                        icon="mdi-content-cut",
                        size=0,
                        click=self.scissors_clicked,
                        active=self.button_active(SegmentationScissorEffect),
                    )

                with VRow():
                    ControlButton(
                        name="Undo",
                        icon="mdi-undo",
                        size=0,
                        click=self.undo_clicked,
                        disabled=(f"!{UndoStack.can_undo_changed.name}",),
                    )
                    ControlButton(
                        name="Redo",
                        icon="mdi-redo",
                        size=0,
                        click=self.redo_clicked,
                        disabled=(f"!{UndoStack.can_redo_changed.name}",),
                    )
                    SegmentationOpacityModeToggleButton(
                        name="Toggle Opacity mode (fill, outline, both)",
                        size=0,
                        click=self.opacity_mode_clicked,
                    )
                with VRow(style="width: 100%", align="center", align_content="center"):
                    Span("2D Opacity", classes="pl-5")
                    VSlider(
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        track_size=2,
                        thumb_size=11,
                        hide_details=True,
                        v_model=SegmentationId.opacity_2d,
                        classes="pr-5",
                    )
                with VRow(style="width: 100%", align="center", align_content="center"):
                    Span("3D Opacity", classes="pl-5")
                    VSlider(
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        track_size=2,
                        thumb_size=11,
                        hide_details=True,
                        v_model=SegmentationId.opacity_3d,
                        classes="pr-5",
                    )
            # 分割列表
            with VContainer(classes="pa-1 fill-height"):
                with VTable(style="width: 100%; height: 300px;", density="compact", fixed_header=True):
                    with Thead():
                        with Tr(style="height: 30px !important; min-height: 20px !important;",):
                            Th((f"Segment List({{{{ {SegmentationId.segments}.length }}}})", ), colspan=2,
                               style="height: 30px !important; min-height: 20px !important; text-align: center;")
                            Th(style="height: 30px !important; min-height: 20px !important; text-align: center;")
                    with Tbody():
                        with Tr(v_for=f"(item, index) in {SegmentationId.segments}",
                                key="item.title",
                                style="height: 30px !important; min-height: 20px !important;",
                                bgcolor=(f"item.props.segment_id==={SegmentationEditor.active_segment_id_changed.name}? '#2a74a2' : ''",),#e3f2fd
                                click=(self.select_segment_id_clicked, "[item.props.segment_id]"), ):
                            with Td(width="40px", style="height: 20px !important; min-height: 20px !important;"):
                                pass
                                VCheckbox(
                                    density="compact",
                                    v_model=("item.props.visibility",),
                                    color=("item.props.color_hex",),
                                    base_color=("item.props.color_hex",),
                                    hide_details=True,
                                    click="$event.stopPropagation();",
                                    update_modelValue=f"""
                                                      trigger('{self.server.trigger_name(self.segment_visibility_toggled)}', [item.props.segment_id, $event]);
                                                      """,
                                    )
                            with Td(style="height: 20px !important; min-height: 20px !important;"):
                                Span("{{item.title}}", classes="pl-2")

    @classmethod
    def button_active(cls, effect_cls: type | None):
        name = effect_cls.__name__ if effect_cls is not None else ""
        return (f"{SegmentationEditor.active_effect_name_changed.name}==='{name}'",)


# slicer.utils.arrayFromSegmentBinaryLabelmap 的实现
def arrayFromSegmentBinaryLabelmap(slicer_app, segmentationNode, segmentId, referenceVolumeNode=None):
    from vtkmodules.vtkCommonCore import vtkStringArray
    from slicer import vtkMRMLSegmentationNode
    import slicer.util
    """Return voxel array of a segment's binary labelmap representation as numpy array.

    :param segmentationNode: source segmentation node.
    :param segmentId: ID of the source segment.
      Can be determined from segment name by calling ``segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)``.
    :param referenceVolumeNode: a volume node that determines geometry (origin, spacing, axis directions, extents) of the array.
      If not specified then the volume that was used for setting the segmentation's geometry is used as reference volume.

    :raises RuntimeError: in case of failure

    Voxels values are copied, therefore changing the returned numpy array has no effect on the source segmentation.
    The modified array can be written back to the segmentation by calling :py:meth:`updateSegmentBinaryLabelmapFromArray`.

    To get voxels of a segment as a modifiable numpy array, you can use :py:meth:`arrayFromSegmentInternalBinaryLabelmap`.
    """
    # Get reference volume
    if not referenceVolumeNode:
        referenceVolumeNode = segmentationNode.GetNodeReference(
            vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole())
        if not referenceVolumeNode:
            raise RuntimeError(
                "No reference volume is found in the input segmentationNode, therefore a valid referenceVolumeNode input is required.")

    # Export segment as vtkImageData (via temporary labelmap volume node)
    segmentIds = vtkStringArray()
    segmentIds.InsertNextValue(segmentId)
    labelmapVolumeNode = slicer_app.scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "__temp__")

    try:
        if not slicer_app.segmentation_logic.ExportSegmentsToLabelmapNode(segmentationNode, segmentIds,
                                                                                 labelmapVolumeNode, referenceVolumeNode):
            raise RuntimeError("Export of segment failed.")
        narray = slicer.util.arrayFromVolume(labelmapVolumeNode)
    finally:
        if labelmapVolumeNode.GetDisplayNode():
            slicer_app.scene.RemoveNode(labelmapVolumeNode.GetDisplayNode().GetColorNode())
        slicer_app.scene.RemoveNode(labelmapVolumeNode)
    return narray


@TrameApp()
class SegmentEditor(VCard):
    def __init__(self, server, slicer_app: SlicerApp, bottom=5, height=500):
        super().__init__(position= "absolute", #"sticky",
                         style=
                         # "position: absolute;"
                         f"bottom: {bottom}px;"
                         f"height: {height}px;"
                         f"width: 100%"
                        )
        self._server = server
        self._slicer_app = slicer_app
        self._volume_node = None
        self._segmentation_node = None

        self._undo_stack = UndoStack(undo_limit=5)
        self.segmentation_editor.set_undo_stack(self._undo_stack)

        self.state.setdefault(SegmentationId.current_segment_id, "")
        self.state.setdefault(SegmentationId.segments, [])
        self.state.setdefault(SegmentationId.is_renaming_segment, False)
        self.state.setdefault(SegmentationId.segment_opacity_mode, SegmentationOpacityEnum.BOTH.value)
        self.state.setdefault(SegmentationId.opacity_2d, 0.5)
        self.state.setdefault(SegmentationId.opacity_3d, 0.5)

        self.connect_segmentation_editor_to_state()
        self.connect_undo_stack_to_state()

        with self:
            # with Template(v_slot_activator="{props}"):
            #     ControlButton(
            #         v_bind="props",
            #         icon="mdi-brush",
            #         name="Segmentation",
            #     )
            with Div("Segment Editor", style="height: 100%", v_if=f"{SegmentationId.segments}.length != 0"):
                with VCard(style="height: 100%; overflow: visible;"), VCardText():
                    self.rename = SegmentationRename(server=server, v_if=(SegmentationId.is_renaming_segment,))
                    self.selection = SegmentSelection(v_else=True)

        self.connect_signals()

    def connect_signals(self):
        self.rename.validate_clicked.connect(self.on_validate_rename)
        self.rename.cancel_clicked.connect(self.on_cancel_rename)

        self.selection.add_segment_clicked.connect(self.on_add_segment)
        self.selection.delete_current_segment_clicked.connect(self.on_delete_current_segment)
        self.selection.start_rename_clicked.connect(self.on_start_rename)
        self.selection.no_tool_clicked.connect(self.on_no_tool)
        self.selection.paint_clicked.connect(self.on_paint)
        self.selection.erase_clicked.connect(self.on_erase)
        self.selection.scissors_clicked.connect(self.on_scissors)
        self.selection.toggle_3d_clicked.connect(self.on_toggle_3d)
        self.selection.segment_visibility_toggled.connect(self.on_toggle_segment_visibility)
        self.selection.opacity_mode_clicked.connect(self.on_toggle_2d_opacity_mode)
        self.selection.undo_clicked.connect(self._undo_stack.undo)
        self.selection.redo_clicked.connect(self._undo_stack.redo)
        self.selection.select_segment_id_clicked.connect(self._select_segment_id)

    def connect_segmentation_editor_to_state(self):
        self.segmentation_editor.segmentation_modified.connect(self._update_segment_properties)
        connect_all_signals_emitting_values_to_state(self.segmentation_editor, self.state)
        self.segmentation_editor.trigger_all_signals()

    def connect_undo_stack_to_state(self):
        connect_all_signals_emitting_values_to_state(self._undo_stack, self.state)
        self._undo_stack.trigger_all_signals()

    @property
    def segmentation_editor(self):
        return self._slicer_app.segmentation_editor

    @property
    def scene(self):
        return self._slicer_app.scene

    def get_selected_segmentation_node_and_segment_id(self):
        segmentation_node = self.get_current_segmentation_node()
        selected_segment_id = self.get_current_segment_id()
        if not selected_segment_id:
            selected_segment_id = self.segmentation_editor.add_empty_segment()

        return segmentation_node, selected_segment_id

    def get_current_segmentation_node(self):
        return self._segmentation_node

    def get_current_segment_id(self) -> str:
        return self.segmentation_editor.active_segment_id

    # def set_current_segment_id(self, segment_id: str | None):
    #     self.state[SegmentationId.current_segment_id] = segment_id

    def _select_segment_id(self, segment_id: str | None):
        self.segmentation_editor.set_active_segment_id(segment_id)
        # self.set_current_segment_id(segment_id)
        self.state[SegmentationId.current_segment_id] = segment_id
        props = self.get_current_segment_properties()
        self.state[SegmentationId.current_segment_name] = props.name

    def get_current_segment_data(self):
        """
        Gets the labelmap array (binary) of the currently selected segment.
        """
        segmentation_node, selected_segment_id = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        mask = arrayFromSegmentBinaryLabelmap(self._slicer_app, segmentation_node, selected_segment_id, self._volume_node)
        seg_data_bool = mask.astype(bool)

        return seg_data_bool

    def get_current_segment_properties(self):
        return self.segmentation_editor.get_segment_properties(self.get_current_segment_id())

    def set_segment_properties(self, segment_properties: SegmentProperties):
        self.segmentation_editor.set_segment_properties(self.get_current_segment_id(), segment_properties)

    @change(StateId.current_volume_node_id)
    def on_volume_changed(self, **_kwargs):
        self._volume_node = get_current_volume_node(self._server, self._slicer_app)
        self.scene.RemoveNode(self._segmentation_node)
        self._segmentation_node = self.segmentation_editor.create_empty_segmentation_node()
        self.segmentation_editor.deactivate_effect()
        self.segmentation_editor.set_active_segmentation(
            self._segmentation_node,
            self._volume_node,
        )
        self.segmentation_editor.set_opacity_mode(
            SegmentationOpacityEnum(self.state[SegmentationId.segment_opacity_mode])
        )
        self.on_add_segment()

    @change(SegmentationEditor.active_segment_id_changed.name)
    def on_current_segment_id_changed(self, **_kwargs):
        self.segmentation_editor.set_active_segment_id(_kwargs[SegmentationEditor.active_segment_id_changed.name])
        # Update opacity for (potentially) new segment
        self.on_opacity_2d_changed()
        self.on_opacity_3d_changed()

    def on_paint(self):
        paint_effect: SegmentationPaintEffect = self.segmentation_editor.set_active_effect_id(SegmentationEffectID.Paint)
        # 暂用下面的方式修改笔刷直径
        for widget in paint_effect.get_widgets():
            paintWidget: SegmentPaintWidget2D = widget
            # vSize = paintWidget.view.render_window().GetScreenSize()[1]
            # relative_brush_size = 5
            paintWidget._brush_diameter_pix = 16  # (relative_brush_size / 100) * vSize

    def on_erase(self):
        erase_effect: SegmentationEraseEffect = self.segmentation_editor.set_active_effect_id(SegmentationEffectID.Erase)
        # 暂用下面的方式修改笔刷直径
        for widget in erase_effect.get_widgets():
            paintWidget: SegmentPaintWidget2D = widget
            # vSize = paintWidget.view.render_window().GetScreenSize()[1]
            # relative_brush_size = 5
            paintWidget._brush_diameter_pix = 16  # (relative_brush_size / 100) * vSize

    def on_scissors(self):
        self.segmentation_editor.set_active_effect_id(SegmentationEffectID.Scissors)

    def on_no_tool(self):
        self.segmentation_editor.deactivate_effect()

    def on_add_segment(self):
        self.segmentation_editor.add_empty_segment()

    def on_delete_current_segment(self):
        self.segmentation_editor.remove_segment(self.get_current_segment_id())

    def on_start_rename(self):
        props = self.get_current_segment_properties()
        if not props:
            return

        self.rename.set_segment_name(props.name)
        self.rename.set_segment_color(props.color_hex)
        self.state[SegmentationId.is_renaming_segment] = True

    def on_validate_rename(self, segment_name, segment_color):
        props = self.get_current_segment_properties()
        if not props:
            return

        props.name = segment_name
        props.color_hex = segment_color
        self.set_segment_properties(props)
        self.on_cancel_rename()

    def on_cancel_rename(self):
        self.state[SegmentationId.is_renaming_segment] = False

    def _update_segment_properties(self):
        self.state[SegmentationId.segments] = [
            {
                "title": segment_properties.name,
                "props": {
                    "segment_id": segment_id,
                    "visibility": self.segmentation_editor.get_segment_visibility(segment_id),
                    **segment_properties.to_dict(),
                },
            }
            for segment_id, segment_properties in self.segmentation_editor.get_all_segment_properties().items()
        ]

    def on_toggle_3d(self):
        self.segmentation_editor.set_surface_representation_enabled(
            not self.segmentation_editor.is_surface_representation_enabled()
        )

    def on_toggle_segment_visibility(self, segment_id, visibility):
        self.segmentation_editor.set_segment_visibility(segment_id, visibility)
        self._update_segment_properties()

    def on_toggle_2d_opacity_mode(self):
        current_opacity_mode = self.state[SegmentationId.segment_opacity_mode]
        new_opacity_mode = SegmentationOpacityEnum(current_opacity_mode).next()
        self.state[SegmentationId.segment_opacity_mode] = new_opacity_mode.value
        self.segmentation_editor.set_opacity_mode(new_opacity_mode)

    @change(SegmentationId.opacity_2d)
    def on_opacity_2d_changed(self, **_kwargs):
        self.segmentation_editor.set_2d_opacity(self.state[SegmentationId.opacity_2d])

    @change(SegmentationId.opacity_3d)
    def on_opacity_3d_changed(self, **_kwargs):
        self.segmentation_editor.set_3d_opacity(self.state[SegmentationId.opacity_3d])

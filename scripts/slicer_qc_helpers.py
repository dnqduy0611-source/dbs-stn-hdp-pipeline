import os

import slicer


def reset_scene(title):
    slicer.mrmlScene.Clear(0)
    slicer.util.mainWindow().setWindowTitle(title)
    layout_manager = slicer.app.layoutManager()
    layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    return layout_manager


def load_volume(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    node = slicer.util.loadVolume(path)
    if node is None:
        raise RuntimeError("Cannot load volume: " + path)
    node.SetName(name)
    return node


def set_slice_backgrounds(layout_manager, red_volume, yellow_volume=None, green_volume=None):
    yellow_volume = yellow_volume or red_volume
    green_volume = green_volume or red_volume
    assignments = {
        "Red": (red_volume, "Axial"),
        "Yellow": (yellow_volume, "Coronal"),
        "Green": (green_volume, "Sagittal"),
    }
    for view_name, (volume, orientation) in assignments.items():
        widget = layout_manager.sliceWidget(view_name)
        widget.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(volume.GetID())
        widget.mrmlSliceNode().SetOrientation(orientation)
        widget.sliceLogic().FitSliceToAll()


def load_segmentation(path, name, color, opacity_2d=0.28, opacity_3d=0.35, create_surface=True):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    node = slicer.util.loadSegmentation(path)
    if node is None:
        raise RuntimeError("Cannot load segmentation: " + path)
    node.SetName(name)
    segmentation = node.GetSegmentation()
    for index in range(segmentation.GetNumberOfSegments()):
        segment_id = segmentation.GetNthSegmentID(index)
        segment = segmentation.GetSegment(segment_id)
        segment.SetName(name if index == 0 else f"{name}_{index + 1}")
        segment.SetColor(*color)
    display = node.GetDisplayNode()
    display.SetVisibility(True)
    display.SetVisibility2DFill(True)
    display.SetVisibility2DOutline(True)
    display.SetOpacity2DFill(opacity_2d)
    display.SetOpacity2DOutline(1.0)
    display.SetOpacity3D(opacity_3d)
    display.SetVisibility3D(create_surface)
    if create_surface:
        node.CreateClosedSurfaceRepresentation()
    return node


def center_on_nodes(layout_manager, nodes):
    bounds = [0.0] * 6
    all_bounds = []
    for node in nodes:
        node.GetRASBounds(bounds)
        if bounds[1] > bounds[0] and bounds[3] > bounds[2] and bounds[5] > bounds[4]:
            all_bounds.append(list(bounds))
    if not all_bounds:
        return
    center = [
        (min(b[0] for b in all_bounds) + max(b[1] for b in all_bounds)) / 2.0,
        (min(b[2] for b in all_bounds) + max(b[3] for b in all_bounds)) / 2.0,
        (min(b[4] for b in all_bounds) + max(b[5] for b in all_bounds)) / 2.0,
    ]
    for view_name in ("Red", "Yellow", "Green"):
        widget = layout_manager.sliceWidget(view_name)
        widget.mrmlSliceNode().JumpSlice(*center)
        widget.sliceLogic().FitSliceToAll()


def show_warning(message):
    slicer.util.showStatusMessage(message, 30000)

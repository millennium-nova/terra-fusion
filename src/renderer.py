import numpy as np
import pyvista as pv
import scipy.ndimage


def _fit_parallel_camera(plotter, points, elevation=35, azimuth=45, padding=1.04):
    """Fit an orthographic camera to points using their screen-space bounds."""
    points = np.asarray(points, dtype=np.float64)

    elev_rad = np.radians(elevation)
    azim_rad = np.radians(azimuth)
    # Unit vector from the focal point towards the camera.
    camera_axis = np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad),
    ])
    forward = -camera_axis

    # These are the actual screen axes after VTK makes view-up perpendicular
    # to the direction of projection.
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    right_coords = points @ right
    up_coords = points @ up
    depth_coords = points @ camera_axis
    right_min, right_max = right_coords.min(), right_coords.max()
    up_min, up_max = up_coords.min(), up_coords.max()
    depth_min, depth_max = depth_coords.min(), depth_coords.max()

    # Unlike grid.center, this is centered after projection. It therefore
    # includes the required vertical/Z correction for the oblique view.
    focal_point = (
        0.5 * (right_min + right_max) * right
        + 0.5 * (up_min + up_max) * up
        + 0.5 * (depth_min + depth_max) * camera_axis
    )

    projected_width = right_max - right_min
    projected_height = up_max - up_min
    viewport_width, viewport_height = plotter.window_size
    aspect = viewport_width / viewport_height

    # parallel_scale is half the visible vertical span. The second term also
    # guarantees that the horizontal span fits the 1200x800 viewport.
    half_visible_height = 0.5 * max(
        projected_height,
        projected_width / aspect,
        1.0,
    )

    depth = depth_max - depth_min
    camera_distance = 0.5 * depth + 2.0 * max(
        projected_width, projected_height, 1.0
    )
    camera_position = focal_point + camera_distance * camera_axis

    plotter.camera_position = [
        tuple(camera_position),
        tuple(focal_point),
        tuple(up),
    ]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = half_visible_height * padding
    plotter.reset_camera_clipping_range()


def render_terrain_3d(texture_uint8, heightmap_int16, output_path, resolution=512):
    """
    Generate a 3D isometric view of the terrain and save it as an image.

    Uses PyVista (VTK) for high-quality GPU-based surface rendering,
    avoiding the grid/lattice artifacts inherent in matplotlib's 3D surface plots.

    Args:
        texture_uint8: (H, W, 3) uint8 numpy array
        heightmap_int16: (H, W) int16 numpy array
        output_path: Path to save the rendered image
        resolution: Target resolution for rendering (downsampled for speed)
    """
    H, W = heightmap_int16.shape

    # Calculate downsampling factor
    step = max(1, min(H, W) // resolution)

    # Downsample
    h_down = heightmap_int16[::step, ::step].astype(np.float32)
    tex_down = texture_uint8[::step, ::step]  # keep uint8 (texture remains untouched/sharp)

    dH, dW = h_down.shape

    # Percentile normalization for height
    p2 = np.percentile(h_down, 2)
    p98 = np.percentile(h_down, 98)
    if p98 > p2:
        h_norm = np.clip((h_down - p2) / (p98 - p2), 0, 1)
    else:
        h_norm = np.zeros_like(h_down)

    # Smooth only the normalized heightmap used by the 3D renderer.
    h_norm = scipy.ndimage.uniform_filter(h_norm, size=13)

    # Scale height for visualization
    z_exaggeration = 0.2
    Z = h_norm * (max(dH, dW) * z_exaggeration)

    # --- Build PyVista StructuredGrid ---
    # Create coordinate arrays: X (columns), Y (rows), Z (height)
    x = np.arange(dW, dtype=np.float32)
    # Invert Y axis so that image row 0 (top) maps to the maximum Y coordinate in 3D
    y = np.arange(dH-1, -1, -1, dtype=np.float32) 
    X, Y = np.meshgrid(x, y)

    # PyVista StructuredGrid expects (nZ, nY, nX) point ordering, flattening in Fortran order.
    grid = pv.StructuredGrid(X, Y, Z)

    # Assign texture as RGB point data
    # Transpose tex_down to (dW, dH, 3) so that C-order flattening matches PyVista's F-order flattening of X/Y/Z
    rgb_flat = tex_down.transpose(1, 0, 2).reshape(-1, 3)
    grid.point_data["RGB"] = rgb_flat

    # --- Render off-screen ---
    # Use a rectangular (landscape) viewport for better framing of the isometric terrain
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
    plotter.set_background("black")

    plotter.add_mesh(
        grid,
        scalars="RGB",
        rgb=True,
        smooth_shading=True,
        show_edges=False,
    )

    # Fit the complete terrain in projected screen space. A 4% margin keeps
    # antialiasing at the silhouette away from the image edge.
    _fit_parallel_camera(
        plotter,
        grid.points,
        elevation=35,
        azimuth=45,
        padding=1.04,
    )

    # Add a directional light from upper-left (azimuth=315°, altitude=45°)
    center = grid.center
    diag = max(dW, dH)
    light = pv.Light(
        position=(
            center[0] - diag,
            center[1] + diag,
            center[2] + diag,
        ),
        focal_point=center,
        intensity=0.7,
    )
    plotter.add_light(light)

    # Ensure ambient lighting so shadows aren't too dark
    ambient_light = pv.Light(light_type="headlight", intensity=0.25)
    plotter.add_light(ambient_light)

    plotter.screenshot(output_path)
    plotter.close()

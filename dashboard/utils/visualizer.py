import os
import torch
import numpy as np
import trimesh
import trimesh.transformations as tf
from mGPT.utils.human_models import get_coord, smpl_x
from pathlib import Path
import pickle

class StreamVisualizer:
    def __init__(self, device="cuda", width=512, height=512):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.width = width
        self.height = height
        self.renderer = None
        self.faces = smpl_x.face
        self._pyrender = None # Lazy load

    def _get_pyrender(self):
        if self._pyrender is None:
            import pyrender
            self._pyrender = pyrender
        return self._pyrender

    def _init_renderer(self):
        if self.renderer is None:
            pr = self._get_pyrender()
            try:
                self.renderer = pr.OffscreenRenderer(
                    viewport_width=self.width, 
                    viewport_height=self.height, 
                    point_size=1.0
                )
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                raise RuntimeError(
                    f"Failed to initialize pyrender. {e}\n"
                    f"Backend selected: {os.environ.get('PYOPENGL_PLATFORM')}\n"
                    "Possible fixes:\n"
                    "1. If OSMesa fails with ImportError, your PyOpenGL may be incomplete.\n"
                    "2. Try running with: xvfb-run -a streamlit run dashboard/app.py\n"
                    f"Full Trace:\n{error_details}"
                ) from e

    def smplx_to_vertices(self, root, body, lhand, rhand, jaw, shape, expr):
        t = root.shape[0]
        with torch.no_grad():
            vertices, _ = get_coord(
                root_pose=root.to(self.device),
                body_pose=body.to(self.device),
                lhand_pose=lhand.to(self.device),
                rhand_pose=rhand.to(self.device),
                jaw_pose=jaw.to(self.device),
                shape=shape.to(self.device),
                expr=expr.to(self.device),
            )
        return vertices.detach().cpu().numpy().reshape(t, -1, 3)

    def render_frame(self, vertices, cam_trans=None, focal=5000.0, mesh_rot_deg=(0, 0, 0)):
        self._init_renderer()
        pr = self._get_pyrender()
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        
        # Rotations
        rx, ry, rz = mesh_rot_deg
        if abs(rx) > 1e-8:
            mesh.apply_transform(tf.rotation_matrix(np.radians(rx), [1, 0, 0]))
        if abs(ry) > 1e-8:
            mesh.apply_transform(tf.rotation_matrix(np.radians(ry), [0, 1, 0]))
        if abs(rz) > 1e-8:
            mesh.apply_transform(tf.rotation_matrix(np.radians(rz), [0, 0, 1]))

        material = pr.MetallicRoughnessMaterial(
            metallicFactor=0.1, roughnessFactor=0.4, alphaMode="OPAQUE",
            baseColorFactor=(0.85, 0.85, 0.9, 1.0),
        )
        mesh_node = pr.Mesh.from_trimesh(mesh, material=material, smooth=True)

        bg_color = np.array([40, 42, 45], dtype=np.float32)
        scene = pr.Scene(ambient_light=(0.15, 0.15, 0.15), bg_color=list(bg_color) + [255])
        scene.add(mesh_node)

        # Camera
        if cam_trans is None:
            min_v, max_v = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
            center = (min_v + max_v) / 2.0
            max_extent = np.max(max_v - min_v)
            distance = (max_extent * focal / min(self.width, self.height)) * 1.2
            cam_trans = np.array([center[0], center[1], center[2] + distance])

        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, 3] = cam_trans
        camera_pose[:3, :3] = np.array([[1, 0, 0],[0, -1, 0], [0, 0, -1]], dtype=np.float32)
        camera = pr.IntrinsicsCamera(fx=focal, fy=focal, cx=self.width/2, cy=self.height/2)
        scene.add(camera, pose=camera_pose)

        # Lights
        scene.add(pr.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=5.0), pose=tf.euler_matrix(-np.pi/4, np.pi/4, 0))
        
        rgb, _ = self.renderer.render(scene, flags=pr.RenderFlags.RGBA)
        return rgb[:, :, :3]

    def cleanup(self):
        if self.renderer:
            self.renderer.delete()
            self.renderer = None

def load_pose_dir(pose_dir):
    p = Path(pose_dir)
    files = sorted([f for f in p.glob("*.pkl")])
    if not files:
        files = sorted(list(p.glob("*.pt")))
    
    if not files:
        raise FileNotFoundError(f"No .pkl or .pt pose files found in {pose_dir}")

    frames = []
    for f in files:
        if f.suffix == ".pkl":
            with open(f, "rb") as fp:
                frames.append(pickle.load(fp))
        else:
            frames.append(torch.load(f, map_location="cpu"))
    
    try:
        root = torch.stack([torch.from_numpy(np.asarray(x["smplx_root_pose"]).reshape(3)) for x in frames])
        body = torch.stack([torch.from_numpy(np.asarray(x["smplx_body_pose"]).reshape(63)) for x in frames])
        lhand = torch.stack([torch.from_numpy(np.asarray(x["smplx_lhand_pose"]).reshape(45)) for x in frames])
        rhand = torch.stack([torch.from_numpy(np.asarray(x["smplx_rhand_pose"]).reshape(45)) for x in frames])
        jaw = torch.stack([torch.from_numpy(np.asarray(x["smplx_jaw_pose"]).reshape(3)) for x in frames])
        expr = torch.stack([torch.from_numpy(np.asarray(x["smplx_expr"]).reshape(10)) for x in frames])
        shape = torch.stack([torch.from_numpy(np.asarray(x["smplx_shape"]).reshape(10)) for x in frames])
    except KeyError as e:
        raise KeyError(f"Missing SMPL-X key {e} in pose file {files[0]}")
    
    return root, body, lhand, rhand, jaw, shape, expr

import os
from pathlib import Path
import pyrender
import trimesh
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser


def load_krt(path):
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name[:-1]] = {
                "intrin": np.array(intrin),
                "dist": np.array(dist),
                "extrin": np.array(extrin),
            }

    return cameras


UINT16_MAX = 65535
SCALE_FACTOR = 1e-1  # corresponds to a representative power of 6.535m with .1 mm resolution


def float32_2_uint16(x):
    float_max = UINT16_MAX * SCALE_FACTOR
    return (x.clip(max=float_max) / SCALE_FACTOR).round().astype(np.uint16)


def uint16_2_float32(x):
    return x.astype(np.float32) * SCALE_FACTOR


OPENCV2OPENGL = np.array([[1., 0., 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", default=Path("data/MULTIFACE"), type=Path)
    parser.add_argument("--subjects", "-s", nargs="*", default=[])
    parser.add_argument("-H", type=int, default=2048)
    parser.add_argument("-W", type=int, default=1334)
    args = parser.parse_args()

    # setting up pyrender scene
    scene = pyrender.Scene()
    renderer = pyrender.OffscreenRenderer(args.W, args.H)

    subjects = args.subjects  # by default preprocesses all subjects
    if not subjects:
        subjects = sorted(os.listdir(args.root))

    # rendering
    for subj in tqdm.tqdm(subjects, desc="Subjects"):
        subj_path = args.root / subj
        krt = load_krt(subj_path / "KRT")
        for seq_path in tqdm.tqdm(sorted((subj_path / "tracked_mesh").iterdir()), desc="Sequences", leave=False):
            for mesh_path in tqdm.tqdm([fp for fp in sorted(seq_path.iterdir()) if fp.name.endswith(".obj")],
                                       desc="Frames",
                                       leave=False):
                mesh = trimesh.load(mesh_path)
                mesh = pyrender.Mesh.from_trimesh(mesh)
                mesh_node = scene.add(mesh)

                for cam_name in sorted(krt.keys()):
                    cam_params = krt[cam_name]
                    cam_intrinsics = cam_params["intrin"]
                    cam_extrinsics = cam_params["extrin"]
                    cam_pose = np.eye(4)
                    cam_pose[:3, :3] = cam_extrinsics[:, :3].T
                    cam_pose[:3, -1:] = -cam_extrinsics[:, :3].T @ cam_extrinsics[:, -1:]
                    cam_pose = cam_pose @ OPENCV2OPENGL

                    cam = pyrender.IntrinsicsCamera(fx=cam_intrinsics[0, 0], fy=cam_intrinsics[1, 1],
                                                    cx=cam_intrinsics[0, -1], cy=cam_intrinsics[1, -1],
                                                    znear=1, zfar=3 * 10e3)
                    cam_node = scene.add(cam, pose=cam_pose)

                    _, depth = renderer.render(scene)
                    alpha = (~(depth == 0)).astype(float)

                    # # visualize result
                    # img_path = subj_path / "images" / seq_path.name / cam_name / (mesh_path.stem + ".png")
                    # img = Image.open(img_path)
                    # plt.figure()
                    # plt.subplot(1, 4, 1)
                    # plt.axis('off')
                    # plt.imshow(img)
                    # plt.subplot(1, 4, 2)
                    # plt.axis('off')
                    # plt.imshow(depth, cmap="turbo")
                    # plt.subplot(1, 4, 3)
                    # plt.axis('off')
                    # plt.imshow(depth, cmap="turbo")
                    # plt.imshow(img, alpha=.3)
                    # plt.subplot(1, 4, 4)
                    # plt.axis('off')
                    # plt.imshow(alpha)
                    # plt.show()

                    # writing output
                    out_dpath = subj_path / "depths" / seq_path.name / cam_name / (mesh_path.stem + ".png")
                    out_apath = subj_path / "masks" / seq_path.name / cam_name / (mesh_path.stem + ".png")
                    out_dpath.parent.mkdir(parents=True, exist_ok=True)
                    out_apath.parent.mkdir(parents=True, exist_ok=True)

                    Image.fromarray(float32_2_uint16(depth)).save(out_dpath)
                    Image.fromarray((alpha * 255).astype(np.uint8)).save(out_apath)

                    # # checking save compression error
                    # depth_saved = uint16_2_float32(np.array(Image.open(out_dpath)))
                    # print(np.max(np.abs(depth_saved-depth)))

                    scene.remove_node(cam_node)
                scene.remove_node(mesh_node)
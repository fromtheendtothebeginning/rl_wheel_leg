"""Isaac Sim 场景辅助：地面 / 灯光 / 相机 / 网格修复。

本机访问不到 NVIDIA 在线资产服务器（omniverse-content-production S3 返回 404），
所以 ``add_default_ground_plane()`` 这类 API 会直接抛
``Could not find assets root folder``。这里的地面、材质、灯光全部用本地 USD prim 手搭。
另外 headless 下没有默认光照，不加灯渲染出来是纯黑。
"""

from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

DEFAULT_UP = Gf.Vec3d(0.0, 0.0, 1.0)


def make_material(stage: Usd.Stage, path: str, color: Gf.Vec3f, roughness: float = 0.6) -> UsdShade.Material:
    """一个最简 UsdPreviewSurface 材质（不依赖任何 MDL / 在线资产）。"""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def world_bbox(stage: Usd.Stage, prim: Usd.Prim) -> tuple[Gf.Vec3d, Gf.Vec3d, float]:
    """返回 (中心, 尺寸, 世界坐标最低 z)。每次新建 BBoxCache，避免读到变换前的缓存。"""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return rng.GetMidpoint(), rng.GetSize(), float(rng.GetMin()[2])


def drop_to_ground(stage: Usd.Stage, prim: Usd.Prim, eps: float = 1e-5) -> float:
    """整体上下平移，让 prim 的最低点贴住 z=0。返回抬升量（米）。

    转换器产出的 USD 原点不一定在地面：URDF 的原点在 base_link（整车顶部），
    轮胎底部会落到 z<0，直接进场景就是陷进地里。
    """
    _, _, zmin = world_bbox(stage, prim)
    if abs(zmin) <= eps:
        return 0.0
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -zmin))
    return -zmin


def add_ground(stage: Usd.Stage, size_m: float = 2.0, color: Gf.Vec3f | None = None) -> str:
    """一块本地地面。

    size_m 别贪大：这些机器人只有十几厘米高，地面铺太大等于把阴影贴图的分辨率
    全浪费在远处，阴影会糊成黑斑。
    """
    path = "/World/ground"
    ground = UsdGeom.Plane.Define(stage, path)
    UsdGeom.Xformable(ground.GetPrim()).AddScaleOp().Set(Gf.Vec3f(size_m, size_m, 1.0))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    mat = make_material(stage, "/World/ground_mat", color or Gf.Vec3f(0.62, 0.64, 0.67))
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(mat)
    return path


def add_lighting(stage: Usd.Stage, dome_intensity: float = 600.0, key_intensity: float = 2500.0) -> None:
    """穹顶光做环境光 + 平行光做主光（不投影）。

    厘米级场景用硬阴影会出 shadow acne，而 UsdLux.ShadowAPI 只暴露 enable/distance、
    没有 bias 可调，所以主光干脆关投影，靠穹顶光把暗部提亮。
    """
    dome = UsdLux.DomeLight.Define(stage, "/World/dome")
    dome.CreateIntensityAttr(dome_intensity)

    key = UsdLux.DistantLight.Define(stage, "/World/key")
    key.CreateIntensityAttr(key_intensity)
    key.CreateAngleAttr(0.5)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 30.0))
    UsdLux.ShadowAPI.Apply(key.GetPrim()).CreateShadowEnableAttr(False)


def add_auto_camera(
    stage: Usd.Stage, target_prim: Usd.Prim, focal_length: float = 28.0, margin: float = 1.9
) -> str:
    """按包围盒自动取景，保证整个物体都在画面里。

    手写机位很容易贴太近，看起来像是"模型散架了"（其实只是视锥切掉了一半）。
    """
    center, size, _ = world_bbox(stage, target_prim)
    dist = max(float(max(size)), 0.1) * margin

    eye = center + Gf.Vec3d(dist * 0.62, -dist * 0.72, dist * 0.45)
    lookat = Gf.Matrix4d()
    lookat.SetLookAt(eye, center, DEFAULT_UP)

    path = "/World/cam"
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal_length)
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(lookat.GetInverse())
    return path


def find_prim(stage: Usd.Stage, name: str) -> Usd.Prim | None:
    """按名字找 prim。两个转换器产出的层级结构不同，别硬编码路径。"""
    return next((p for p in stage.Traverse() if p.GetName() == name), None)


def repair_robot_meshes(
    stage: Usd.Stage,
    material_path: str = "/World/robot_mat",
    color: Gf.Vec3f | None = None,
    hide_guides: bool = True,
) -> dict[str, int]:
    """修视觉网格，并隐藏碰撞体。

    SolidWorks 导出的 STL 是薄板开口壳体：既没法线也没材质，RTX 自己猜法线时
    薄板内侧会朝里，渲成纯黑斑块。这里用 trimesh 算朝外的平面法线写回（CAD 件就该
    是硬边，所以用 faceVarying + 面法线），再设 doubleSided、绑一个材质。
    注意：开放壳体判断不了内外，黑斑只能减轻、不能根除。

    碰撞体（凸分解产物）的 purpose 是 ``guide``，没有材质，不隐藏会渲成黑块。
    """
    import trimesh

    mat = make_material(stage, material_path, color or Gf.Vec3f(0.88, 0.89, 0.90))
    stats = {"visual": 0, "normals_fixed": 0, "guides_hidden": 0, "failed": 0}

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        purpose = str(UsdGeom.Imageable(prim).GetPurposeAttr().Get())

        if purpose == "guide":
            if hide_guides:
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set("invisible")
                stats["guides_hidden"] += 1
            continue
        if purpose != "default":
            continue

        mesh = UsdGeom.Mesh(prim)
        UsdGeom.Gprim(prim).CreateDoubleSidedAttr().Set(True)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
        stats["visual"] += 1

        try:
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            if counts.size == 0 or counts.min() != 3 or indices.size == 0:
                continue
            tri = trimesh.Trimesh(vertices=points, faces=indices.reshape(-1, 3), process=False)
            tri.fix_normals()
            mesh.GetNormalsAttr().Set(np.repeat(tri.face_normals.astype(np.float32), 3, axis=0))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            stats["normals_fixed"] += 1
        except Exception as exc:  # noqa: BLE001 - 单个网格失败不该中断整个场景
            stats["failed"] += 1
            print(f"      法线修复失败 {prim.GetName()}: {exc}", flush=True)

    return stats


def _anchor_world(stage: Usd.Stage, body_rel, local_pos) -> Gf.Vec3d | None:
    targets = body_rel.GetTargets()
    if not targets:
        return None
    body = stage.GetPrimAtPath(targets[0])
    if not body:
        return None
    xform = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return xform.Transform(Gf.Vec3d(*local_pos))


def closed_loop_residual(stage: Usd.Stage) -> tuple[float, int]:
    """闭链残差（米）+ 参与统计的闭链条数。

    必须把 joint 两端的**锚点**（localPos0/localPos1）各自换算到世界再量距离。
    比较两个 body 的原点距离是错的 —— 两个铰接连杆的原点本来就不重合，那数字没有意义。
    """
    worst, count = 0.0, 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.SphericalJoint):
            continue
        joint = UsdPhysics.SphericalJoint(prim)
        p0 = _anchor_world(stage, joint.GetBody0Rel(), joint.GetLocalPos0Attr().Get() or (0, 0, 0))
        p1 = _anchor_world(stage, joint.GetBody1Rel(), joint.GetLocalPos1Attr().Get() or (0, 0, 0))
        if p0 is None or p1 is None:
            continue
        worst = max(worst, (p0 - p1).GetLength())
        count += 1
    return worst, count

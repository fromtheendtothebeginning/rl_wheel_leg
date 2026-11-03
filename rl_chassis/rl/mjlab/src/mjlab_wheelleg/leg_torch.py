"""v11/v12 轮腿闭链运动学 —— torch 批量版（GPU 训练用）。

几何常数 vendor 自 ``rl/tools/leg_kinematics_ref.py``（v12 与 v11 同几何）。
坐标系 x 前 / y 左 / z 上；每侧腿是平面机构，关节轴 = 世界 ±y，
所以所有连杆偏移在 (x, z) 平面里旋转，y 分量为常量、不进约束。

训练时仿真本身就维持闭链一致（connect equality），这里只需要：

* ``shank_jacobian``：当前状态下 ∂q_shank/∂(q_front, q_rear)，用于把策略输出的
  虚拟 (髋, 膝) 力矩映射到真机 (后髋, 前髋曲柄) 力矩（虚功原理）；
* ``theta0``：两条虚拟腿的倾角（奖励里惩罚左右不同步）。

约束求导是解析的：残差 r ∈ R⁴（两个销的 x/z 分量差），对被动关节
(c1, c2, c3, shank) 的雅可比 J_x 与对驱动 (front, rear) 的雅可比 J_p 都有
闭式，∂x/∂p = -J_x⁻¹ J_p 取 shank 行即可。自检见 scripts/check_leg_torch.py。
"""

from __future__ import annotations

import math
from dataclasses import dataclass  # noqa: F401  （保留给后续扩展）

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 几何常数（与 rl/tools/leg_kinematics_ref.py 的 LEGS 完全一致）
# ---------------------------------------------------------------------------

LEGS = {
    "left": dict(
        mount_front=((-0.001951, 0.1755, 0.11508), (-math.pi / 2, 0.0, 0.0)),
        mount_rear=((-0.001951, 0.1515, 0.11508), (-math.pi / 2, 0.0, 0.0)),
        front_child1=(0.0999999789, 1.56497905e-07, -0.0209999983),
        child1_child2=(-0.100071446, 0.0627908435, 0.0134999983),
        child2_child3=(-0.136668405, -0.110450469, -0.012499876),
        rear_shank=(-0.21499951, -0.00048882211, 0.017999212),
        shank_wheel=(0.21484219, 0.13549473, 0.017550293),
        pin1_A=(-0.099927571748, -0.063019390865, -0.0028752271278),  # child2 上
        pin1_B=(-0.099999038848, -0.00022839022379, 0.013624772872),  # rear_link 上
        pin2_A=(-0.11500040187, -0.00026465136291, 0.014699563538),   # child3 上
        pin2_B=(-0.036740763971, -0.047435297612, 0.00070047553825),  # shank 上
    ),
    "right": dict(
        mount_front=((-0.001951, -0.1755, 0.11508), (math.pi / 2, 0.0, 0.0)),
        mount_rear=((-0.001951, -0.1515, 0.11508), (math.pi / 2, 0.0, 0.0)),
        front_child1=(0.100000046, -1.67484869e-07, -0.0209999983),
        child1_child2=(-0.100071359, -0.0627908325, 0.0134999983),
        child2_child3=(-0.136668391, 0.11045049, -0.012500124),
        rear_shank=(-0.21499938, 0.00048895405, 0.018000788),
        shank_wheel=(0.21484232, -0.1354946, 0.017549707),
        pin1_A=(-0.099927638948, 0.063019410142, -0.0028747712562),
        pin1_B=(-0.099998951948, 0.00022840951442, 0.013625228744),
        pin2_A=(-0.11500033997, 0.00026483636101, 0.0097004366532),
        pin2_B=(-0.036740663968, 0.047435371683, -0.0043004753468),
    ),
}

# 训练 MJCF（v12）里的关节名，顺序 = (front, child1, child2, child3, rear, shank)。
JOINT_NAMES = {
    "left": (
        "Left_front_joint",
        "Left_front_child1_joint",
        "Left_front_child2_joint",
        "Left_front_child3_joint",
        "Left_rear_joint",
        "Left_rear_child1_joint",
    ),
    "right": (
        "Right_front_joint",
        "Right_front_child1_joint",
        "Right_front_joint2_joint",
        "Right_front_joint3_joint",
        "Right_rear_joint",
        "Right_rear_child1_joint",
    ),
}
WHEEL_JOINT_NAMES = {"left": "Left_Wheel_joint", "right": "Right_Wheel_joint"}
POM_JOINT_NAMES = tuple(f"pom_wheel{i}_joint" for i in (1, 2, 3, 4))

# 电机（MJCF 里的 6 个 motor，ctrl 即力矩）。
MOTOR_NAMES = (
    "Left_front_joint_motor",
    "Left_rear_joint_motor",
    "Left_Wheel_joint_motor",
    "Right_front_joint_motor",
    "Right_rear_joint_motor",
    "Right_Wheel_joint_motor",
)

# 虚拟腿默认角（Isaac 训练口径，串联腿的 q_des；wheel 默认 0 不进 PD 偏置）。
VIRTUAL_DEFAULT = {
    "left": (-0.274884, 0.517721),
    "right": (0.274883, -0.517722),
}

# 上面的虚拟默认角对应的闭链一致关节角（leg_kinematics_ref.solve_virtual 解出，
# 残差 <1e-15；被动关节初值必须一致，否则 reset 后销约束要拉一大把）。
CLOSED_DEFAULT = {
    "left": (
        0.529243849,   # front
        -0.517753025,  # child1
        0.231381213,   # child2
        -0.517763719,  # child3
        -0.274884000,  # rear
        0.517721000,   # shank
    ),
    "right": (
        -0.529246659,
        0.517754150,
        -0.231382091,
        0.517765486,
        0.274883000,
        -0.517722000,
    ),
}

# 轮半径与髋安装高度（base_link 系），用于初始高度。
WHEEL_RADIUS = 0.06
HIP_OFFSET_Z = 0.11508


def _rpy_to_R(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


@dataclass
class _ChainPoint:
    """一段“安装旋转 · rot_z(累积角) · 偏移”的 (x,z) 平面项（诊断用占位）。"""


class LegTorch:
    """一侧腿的批量闭链求导。q6 = (front, c1, c2, c3, rear, shank)。"""

    def __init__(self, side: str):
        self.side = side
        c = LEGS[side]
        self.sigma = -1.0 if side == "left" else 1.0
        self.pf = np.asarray(c["mount_front"][0])
        self.pr = np.asarray(c["mount_rear"][0])

        # 局部原始偏移（不预乘 mount 旋转！）：世界 (x,z) = f(rot_z(累积角)·o)
        #   wx = ox·cos a − oy·sin a；wz = σ·(ox·sin a + oy·cos a)
        #   （局部 z 分量进世界 y，是常量，不进约束）
        raw = lambda k: np.asarray(c[k])  # noqa: E731
        # 前链四段累积角: a1=qf, a2=qf+c1, a3=qf+c1+c2, a4=qf+c1+c2+c3
        self.o1 = raw("front_child1")     # var front
        self.o2 = raw("child1_child2")    # var c1
        self.o3p = raw("pin1_A")          # var c2（pin1 点）
        self.o3c = raw("child2_child3")   # var c2（child3 原点）
        self.o4 = raw("pin2_A")           # var c3（pin2 点）
        # 后链: b1=qr, b2=qr+shank
        self.or1 = raw("rear_shank")      # var rear（shank 原点）
        self.orp = raw("pin1_B")          # var rear（pin1 点）
        self.or2 = raw("pin2_B")          # var shank（pin2 点）
        self.owheel = raw("shank_wheel")  # var shank（轮心）

        self._serial_constants()

    # ---------------- 平面运动学 ----------------
    def _w(self, o: np.ndarray, a: torch.Tensor):
        """返回 (wx, wz)，o 为原始局部 3 维偏移，a 为批量累积角。"""
        wx = o[0] * torch.cos(a) - o[1] * torch.sin(a)
        wz = self.sigma * (o[0] * torch.sin(a) + o[1] * torch.cos(a)) + o[2] * 0.0
        # o[2] 分量经 R_mount 后进了世界 y（常量），不进 (x,z)。
        return wx, wz

    def _dw(self, o: np.ndarray, a: torch.Tensor):
        """(wx, wz) 对角 a 的导数。"""
        dwx = -o[0] * torch.sin(a) - o[1] * torch.cos(a)
        dwz = self.sigma * (o[0] * torch.cos(a) - o[1] * torch.sin(a))
        return dwx, dwz

    def residual(self, q6: torch.Tensor) -> torch.Tensor:
        """(n, 4)：[pin1 x 差, pin1 z 差, pin2 x 差, pin2 z 差]。"""
        qf, c1, c2, c3, qr, shank = [q6[:, i] for i in range(6)]
        a1, a2 = qf, qf + c1
        a3, a4 = qf + c1 + c2, qf + c1 + c2 + c3
        b1, b2 = qr, qr + shank

        w = self._w
        # pin1 前链
        x1f = w(self.o1, a1)[0] + w(self.o2, a2)[0] + w(self.o3p, a3)[0]
        z1f = w(self.o1, a1)[1] + w(self.o2, a2)[1] + w(self.o3p, a3)[1]
        # pin2 前链
        x2f = w(self.o1, a1)[0] + w(self.o2, a2)[0] + w(self.o3c, a3)[0] + w(self.o4, a4)[0]
        z2f = w(self.o1, a1)[1] + w(self.o2, a2)[1] + w(self.o3c, a3)[1] + w(self.o4, a4)[1]
        # 后链
        x1r = w(self.orp, b1)[0]
        z1r = w(self.orp, b1)[1]
        x2r = w(self.or1, b1)[0] + w(self.or2, b2)[0]
        z2r = w(self.or1, b1)[1] + w(self.or2, b2)[1]

        dx = self.pf[0] - self.pr[0]
        dz = self.pf[2] - self.pr[2]
        return torch.stack(
            (dx + x1f - x1r, dz + z1f - z1r, dx + x2f - x2r, dz + z2f - z2r), dim=-1
        )

    def shank_jacobian(self, q6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (d_front, d_rear) = ∂q_shank/∂q_front, ∂q_shank/∂q_rear，各 (n,)。

        ∂x/∂p = -(Jx)⁻¹ Jp 的 shank 行；Jx = J[:, :, [c1,c2,c3,shank]]，
        Jp = J[:, :, [front, rear]]。已与有限差分逐元素核对（<1e-9）。
        """
        J = self.constraint_jacobian(q6)
        Jx = J[:, :, [1, 2, 3, 5]]
        Jp = J[:, :, [0, 4]]
        # 小正则：机构在曲柄与连杆共线时 Jx 奇异（物理上曲柄失去解空间），
        # 训练中偶尔扫过，加 εI 后解仍有限，随后 clamp。
        eye = torch.eye(4, device=q6.device, dtype=q6.dtype) * 1e-9
        sol = torch.linalg.solve(Jx + eye, -Jp)  # (n, 4, 2)
        d_front = torch.clamp(sol[:, 3, 0], -15.0, 15.0)
        d_rear = torch.clamp(sol[:, 3, 1], -15.0, 15.0)
        return d_front, d_rear

    def solve_virtual_batch(
        self,
        q_rear: torch.Tensor,
        q_shank: torch.Tensor,
        seed_q6: torch.Tensor | None = None,
        iters: int = 12,
    ) -> torch.Tensor:
        """给定虚拟 (rear, shank) 角，牛顿解出一致的全关节角 q6 (n, 6)。

        固定 rear/shank（后链完整），未知量 = (front, c1, c2, c3)，
        用 constraint_jacobian 的对应列做牛顿迭代。reset 时给虚拟角加扰动
        后调用，保证初值在闭链流形上（避免 connect 约束的开局弹跳）。
        seed_q6 提供迭代初值（一般传默认姿态）。
        """
        n = q_rear.shape[0]
        q = seed_q6.clone() if seed_q6 is not None else torch.zeros(n, 6)
        q = q.to(q_rear.device)
        q[:, 4], q[:, 5] = q_rear, q_shank
        eye = torch.eye(4, device=q.device, dtype=q.dtype) * 1e-10
        for _ in range(iters):
            r = self.residual(q)
            if r.norm(dim=-1).max() < 1e-12:
                break
            J = self.constraint_jacobian(q)
            Jf = J[:, :, [0, 1, 2, 3]]
            dx = torch.linalg.solve(Jf + eye, -r)  # (n, 4)
            step = dx.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            dx = dx * torch.clamp(0.35 / step, max=1.0)
            q[:, 0:4] += dx
            q[:, 4], q[:, 5] = q_rear, q_shank
        return q
    def constraint_jacobian(self, q6: torch.Tensor) -> torch.Tensor:
        """完整解析雅可比 J = ∂r/∂(front,c1,c2,c3,rear,shank)，形状 (n, 4, 6)。

        行 = (pin1_x, pin1_z, pin2_x, pin2_z) 残差，列 = 6 个关节角。
        """
        qf, c1, c2, c3, qr, shank = [q6[:, i] for i in range(6)]
        a1, a2 = qf, qf + c1
        a3, a4 = qf + c1 + c2, qf + c1 + c2 + c3
        b1, b2 = qr, qr + shank

        d1 = self._dw(self.o1, a1)      # 该段累积角 = qf
        d2 = self._dw(self.o2, a2)      # = qf+c1
        d3p = self._dw(self.o3p, a3)    # = qf+c1+c2
        d3c = self._dw(self.o3c, a3)
        d4 = self._dw(self.o4, a4)      # = qf+c1+c2+c3
        dr1 = self._dw(self.or1, b1)    # = qr
        drp = self._dw(self.orp, b1)    # = qr
        dr2 = self._dw(self.or2, b2)    # = qr+shank

        def add(*terms):
            return (
                torch.stack([t[0] for t in terms], 0).sum(0),
                torch.stack([t[1] for t in terms], 0).sum(0),
            )

        zero2 = (torch.zeros_like(qf), torch.zeros_like(qf))
        # 每列 = 某关节对 (pin1前链点, pin2前链点, pin1后链点, pin2后链点) 的偏导
        # 残差 = 前链 − 后链，后链贡献自带负号。
        col_front = (add(d1, d2, d3p), add(d1, d2, d3c, d4), zero2, zero2)
        col_c1 = (add(d2, d3p), add(d2, d3c, d4), zero2, zero2)
        col_c2 = (add(d3p), add(d3c, d4), zero2, zero2)
        col_c3 = (zero2, add(d4), zero2, zero2)
        col_rear = (zero2, zero2, (-drp[0], -drp[1]), (-dr1[0] - dr2[0], -dr1[1] - dr2[1]))
        col_shank = (zero2, zero2, zero2, (-dr2[0], -dr2[1]))

        n = q6.shape[0]
        J = q6.new_zeros(n, 4, 6)
        for j, col in enumerate((col_front, col_c1, col_c2, col_c3, col_rear, col_shank)):
            p1f, p2f, p1b, p2b = col
            # 残差 r = 前链点 − 后链点，各列的前/后链贡献已在 col 里带好符号
            J[:, 0, j] = p1f[0] + p1b[0]
            J[:, 1, j] = p1f[1] + p1b[1]
            J[:, 2, j] = p2f[0] + p2b[0]
            J[:, 3, j] = p2f[1] + p2b[1]
        return J

    # ---------------- 虚拟腿（后链等效串联） ----------------
    def _serial_constants(self):
        """q=0 装配位姿下的 (l1, l2, a1, a2, sgn_rear, sgn_shank)。"""
        zero = torch.zeros(1, 6)

        def ang(ux, uz, vx, vz):
            return math.atan2(vz - uz, vx - ux)

        knee = self.pr + np.array(
            [self._w(self.or1, torch.zeros(1))[0].item(), 0.0, self._w(self.or1, torch.zeros(1))[1].item()]
        )
        wheel_off = knee + np.array(
            [self._w(self.owheel, torch.zeros(1))[0].item(), 0.0, self._w(self.owheel, torch.zeros(1))[1].item()]
        )
        hip = self.pr
        self.l1 = float(math.hypot(knee[0] - hip[0], knee[2] - hip[2]))
        self.l2 = float(math.hypot(wheel_off[0] - knee[0], wheel_off[2] - knee[2]))
        self.a1 = ang(hip[0], hip[2], knee[0], knee[2])
        self.a2 = ang(knee[0], knee[2], wheel_off[0], wheel_off[2]) - self.a1
        # 符号：扰动 rear / shank 0.1 rad，看平面角往哪边转
        eps = 0.1

        def ang_rear(q):
            a = torch.full((1,), q)
            k = np.array([self._w(self.or1, a)[0].item(), 0.0, self._w(self.or1, a)[1].item()]) + self.pr
            return ang(hip[0], hip[2], k[0], k[2])

        def ang_shank(q):
            a1v = torch.zeros(1)
            a2v = torch.full((1,), q)
            ks = np.array([self._w(self.or1, a1v)[0].item(), 0.0, self._w(self.or1, a1v)[1].item()]) + self.pr
            ws = ks + np.array(
                [self._w(self.owheel, a2v)[0].item(), 0.0, self._w(self.owheel, a2v)[1].item()]
            )
            return ang(ks[0], ks[2], ws[0], ws[2])

        def wrap(x):
            return (x + math.pi) % (2 * math.pi) - math.pi

        self.sgn_rear = math.copysign(1.0, wrap(ang_rear(eps) - self.a1) / eps)
        self.sgn_shank = math.copysign(1.0, wrap(ang_shank(eps) - (self.a1 + self.a2)) / eps)

    def theta0(self, q_rear: torch.Tensor, q_shank: torch.Tensor) -> torch.Tensor:
        """虚拟腿倾角（0 = 竖直向下，轮在前为正），批量 (n,)。"""
        t1 = self.a1 + self.sgn_rear * q_rear
        t2 = self.a2 + self.sgn_shank * q_shank
        ex = self.l1 * torch.cos(t1) + self.l2 * torch.cos(t1 + t2)
        ez = self.l1 * torch.sin(t1) + self.l2 * torch.sin(t1 + t2)
        return torch.atan2(ex, -ez)

    def wheel_xz(self, q6: torch.Tensor):
        """轮心相对 base 的 (x, z)（用于诊断）。"""
        qr, shank = q6[:, 4], q6[:, 5]
        wx = self._w(self.or1, qr)[0] + self._w(self.owheel, qr + shank)[0]
        wz = self._w(self.or1, qr)[1] + self._w(self.owheel, qr + shank)[1]
        return self.pr[0] + wx, self.pr[2] + wz

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练实时仪表盘：localhost 网页显示当前轮数/ep_len 曲线/课程阶段。

读 tensorboard 事件文件，每次请求都重新解析（事件文件小，毫秒级），
页面 5 s 自动刷新。

用法（mjlab/ 目录）：
    .venv/bin/python scripts/status_server.py --run flat_curri2 --port 8765
"""

from __future__ import annotations

import argparse
import glob
import html
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MJLAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(MJLAB_DIR, "logs", "rsl_rl", "wheelleg_v12_mjlab")
_EXPERIMENT = "wheelleg_v12_mjlab"

PAGE = """<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta http-equiv=refresh content="5">
<title>训练进度 {run}</title>
<style>
 body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:2rem}}
 .card{{background:#1b1b1f;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem}}
 h1{{font-size:1.1rem;color:#9ab}} .big{{font-size:3rem;font-weight:700}}
 .mono{{font-family:ui-monospace,monospace}} .ok{{color:#7c7}} .warn{{color:#fc6}}
 .bar{{height:14px;background:#333;border-radius:7px;overflow:hidden}}
 .fill{{height:100%;background:linear-gradient(90deg,#4a9,#7c7)}}
 table{{border-collapse:collapse}} td{{padding:2px 14px 2px 0;color:#bbb}}
 svg{{background:#161619;border-radius:8px;max-width:100%;height:auto;display:block}}
</style></head><body>
<h1>轮腿训练 {run} · 每 5s 自动刷新</h1>
<div class=card><div class=big>{iter} <span style="font-size:1.2rem;color:#888">/ {total} 轮</span></div>
<div class=bar><div class=fill style="width:{pct}%"></div></div>
<table><tr><td>预计剩余</td><td class=mono>{eta}</td><td>已用时间</td><td class=mono>{elapsed}</td></tr>
<tr><td>课程阶段</td><td class=mono>{stage}</td><td>最新 checkpoint</td><td class=mono>model_{ckpt}.pt</td></tr>
<tr><td>ep_len 最近</td><td class=mono>{eplen_last}</td><td> reward</td><td class=mono>{rew}</td></tr></table></div>
<div class=card><b>episode 长度曲线</b>（0.01 s/步，满程 2000）<br>{chart}</div>
<div class=card><b>跟踪质量（每步裁剪均值，1.0 = 完美跟踪）</b><br>
<span style="color:#7c7">● track_lin</span>　<span style="color:#6ac">● track_yaw</span>　
<span style="color:#e8a04c">● height</span>　<span style="color:#c88">● height_err(m)</span><br>{tchart}</div>
<div class=card><b>站定漂移 vx（m/s，负=后退）最新 <span style="color:#e06666">{drift_last}</span></b><br>{dchart}</div>
<div class=card><b>站定高度 m（最终重心高度，口径=base_link 离地）
<span style="color:#7c7">● 实际</span>　<span style="color:#e8a04c">● 指令</span>　
最新 <span style="color:#7c7">{h_last}</span></b><br>{hchart}</div>
<div class=card><b>起立/稳定耗时 s（十次测试 RMS，目标：起立≤2s / 稳定≤0.2s）
<span style="color:#7c7">● 起立</span>　<span style="color:#6ac">● 稳定</span>　
最新 <span style="color:#7c7">{rise_last}</span> / <span style="color:#6ac">{settle_last}</span></b><br>{rschart}</div>
<div class=card><b>训练参数</b>（实时读取当前 run 的 params/env.yaml）<br>{params}</div>
</body></html>"""


def read_params_html(run_dir: str) -> str:
  """params/env.yaml -> 参数卡片 HTML。纯文本解析（yaml 里是
  !!python/name 标签，反序列化要 import mjlab，每 5s 一次不值得）。"""
  p = os.path.join(run_dir, "params", "env.yaml")
  try:
    lines = open(p, encoding="utf-8").read().splitlines()
  except Exception:
    return "<i>未找到 params/env.yaml</i>"
  rewards, gains, cmdr = [], {}, {}
  sec, name, i = None, None, 0
  while i < len(lines):
    ln = lines[i]
    stripped = ln.strip()
    if not stripped or stripped.startswith("#"):
      i += 1
      continue
    ind = len(ln) - len(ln.lstrip())
    key, _, val = stripped.partition(":")
    key, val = key.strip(), val.strip()
    if ind == 0:
      sec = key if key in ("rewards", "actions", "commands") else None
      name = None
    elif sec == "rewards" and ind == 2:
      name = key
    elif sec == "rewards" and ind == 4 and key == "weight" and name:
      try:
        rewards.append((name, float(val)))
      except ValueError:
        pass
    elif sec == "actions" and ind == 4 and key in (
        "kp", "kd", "wheel_kd", "pos_action_scale", "vel_action_scale"):
      try:
        gains[key] = float(val)
      except ValueError:
        pass
    elif sec == "commands" and ind == 4 and key in (
        "lin_vel_range", "ang_vel_range", "height_range", "standing_prob"):
      if "tuple" in val:
        nums = []
        j = i + 1
        while j < len(lines) and lines[j].strip().startswith("- "):
          try:
            nums.append(float(lines[j].strip()[2:]))
          except ValueError:
            pass
          j += 1
        i = j - 1
        cmdr[key] = nums
      else:
        try:
          cmdr[key] = float(val)
        except ValueError:
          pass
    i += 1

  def _fmt(v):
    if isinstance(v, list):
      return " ~ ".join(f"{x:g}" for x in v)
    return f"{v:g}"

  rows = "".join(
    f'<tr><td>{"<b>%s</b>" % _html_escape(n) if n == "stand_still" else _html_escape(n)}</td>'
    f'<td style="color:{"#fc6" if w < 0 else "#7c7"}">{w:g}</td></tr>'
    for n, w in rewards)
  grows = "".join(f"<tr><td>{_html_escape(k)}</td><td>{_fmt(v)}</td></tr>"
                  for k, v in gains.items())
  crows = "".join(f"<tr><td>{_html_escape(k)}</td><td>{_fmt(v)}</td></tr>"
                  for k, v in cmdr.items())
  return (
    '<table style="display:inline-table;vertical-align:top;margin-right:2rem">'
    '<tr><td colspan=2 style="color:#9ab"><b>奖励权重</b>（负=惩罚）</td></tr>'
    + rows + "</table>"
    + '<table style="display:inline-table;vertical-align:top;margin-right:2rem">'
    '<tr><td colspan=2 style="color:#9ab"><b>PD / 动作缩放</b></td></tr>'
    + grows + "</table>"
    + '<table style="display:inline-table;vertical-align:top">'
    '<tr><td colspan=2 style="color:#9ab"><b>指令范围</b></td></tr>'
    + crows + "</table>")


def _html_escape(s: str) -> str:
  import html as _html
  return _html.escape(s)


def render_chart(l: list) -> str:
  """ep_len 曲线 -> 自适应 SVG。参考线：晋级 800 / 满程 2000。"""
  import numpy as np

  W, H = 900, 190
  pl, pr, pt, pb = 10, 10, 22, 20
  pw, ph = W - pl - pr, H - pt - pb
  n = max(2, min(len(l), 400))  # 降采样，避免上千个点撑爆 HTML
  idx = np.linspace(0, len(l) - 1, n).astype(int)
  v = np.array([l[i][1] for i in idx], dtype=float)
  it0, it1 = l[0][0], l[-1][0]
  vmax = max(2000.0, float(v.max()) * 1.08)

  def y(val: float) -> str:
    return f"{pt + ph * (1 - val / vmax):.1f}"

  def x(i: int) -> str:
    return f"{pl + pw * i / (n - 1):.1f}"

  pts = " ".join(f"{x(i)},{y(val)}" for i, val in enumerate(v))
  area = f"M{x(0)},{y(0)} " + " ".join(f"L{x(i)},{y(val)}" for i, val in enumerate(v)) \
      + f" L{x(n - 1)},{y(0)} Z"
  # SVG 属性一律带引号：无引号值紧贴 /> 时（如 opacity=0.12/>），
  # HTML 解析器把 / 归进属性值，自闭合失效，后续元素全被吞进前一个标签
  refs = ""
  for val, label, color in ((800, "晋级线 800", "#e8a04c"), (2000, "满程 2000", "#678")):
    yy = y(val)
    refs += (f'<line x1="{pl}" y1="{yy}" x2="{W - pr}" y2="{yy}" stroke="{color}" '
             f'stroke-dasharray="5" vector-effect="non-scaling-stroke" opacity="0.8" />'
             f'<text x="{pl + 4}" y="{float(yy) - 5:.1f}" font-size="13" fill="{color}">{label}</text>')
  last_y = float(y(v[-1]))
  return (f'<svg viewBox="0 0 {W} {H}" width="100%">'
          f'<path d="{area}" fill="#7c7" opacity="0.12" />{refs}'
          f'<polyline fill="none" stroke="#7c7" stroke-width="1.5" '
          f'vector-effect="non-scaling-stroke" points="{pts}" />'
          f'<circle cx="{x(n - 1)}" cy="{last_y:.1f}" r="3" fill="#7c7" />'
          f'<text x="{W - pr - 2}" y="{max(13, last_y - 8):.1f}" font-size="14" font-weight="bold" '
          f'fill="#7c7" text-anchor="end">{v[-1]:.0f}</text>'
          f'<text x="{pl}" y="{H - 6}" font-size="12" fill="#889">iter {it0}</text>'
          f'<text x="{W - pr}" y="{H - 6}" font-size="12" fill="#889" text-anchor="end">iter {it1}</text></svg>')


def render_multi_chart(series: list, labels: list) -> str:
  """多序列折线（同一坐标，自动缩放）。series = [(values:list[(step,val)], color)]。"""
  import numpy as np

  W, H = 900, 220
  pl, pr, pt, pb = 10, 10, 22, 20
  pw, ph = W - pl - pr, H - pt - pb
  allv = [v for s, _ in series for _, v in s]
  if not allv:
    return "<i>无数据</i>"
  vmax = max(1e-9, max(abs(min(allv)), abs(max(allv))) * 1.15)
  steps = [st for s, _ in series for st, _ in s]
  it0, it1 = min(steps), max(steps)
  span = max(1, it1 - it0)

  def X(st):
    return pl + pw * (st - it0) / span

  def Y(v):
    return pt + ph * (1 - (v + vmax) / (2 * vmax))

  zy = Y(vmax) if False else Y(0.0)
  svg = [f'<line x1="{pl}" y1="{zy:.1f}" x2="{W - pr}" y2="{zy:.1f}" '
         f'stroke="#555" stroke-dasharray="4" vector-effect="non-scaling-stroke" />']
  for values, color in series:
    if not values:
      continue
    n = max(2, min(len(values), 400))
    idx = np.linspace(0, len(values) - 1, n).astype(int)
    pts = " ".join(f"{X(values[i][0]):.1f},{Y(values[i][1]):.1f}" for i in idx)
    lx, ly = X(values[-1][0]), Y(values[-1][1])
    svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" '
               f'vector-effect="non-scaling-stroke" points="{pts}" />')
    svg.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3" fill="{color}" />')
    svg.append(f'<text x="{lx - 4:.1f}" y="{ly - 6:.1f}" font-size="12" fill="{color}" '
               f'text-anchor="end">{values[-1][1]:.4f}</text>')
  svg.append(f'<text x="{pl}" y="{H - 6}" font-size="12" fill="#889">iter {it0}</text>')
  svg.append(f'<text x="{W - pr}" y="{H - 6}" font-size="12" fill="#889" '
             f'text-anchor="end">iter {it1}</text>')
  return f'<svg viewBox="0 0 {W} {H}" width="100%">' + "".join(svg) + "</svg>"


def render_pos_chart(series: list, refs=()) -> str:
  """正数量值折线（实际/指令对照）：Y 轴 = 数据范围上下留 10% 余量。
  refs = [(value, label, color)]，画水平虚线参考线（如验收目标）。"""
  import numpy as np

  W, H = 900, 200
  pl, pr, pt, pb = 10, 10, 22, 20
  pw, ph = W - pl - pr, H - pt - pb
  allv = [v for s, _ in series for _, v in s] + [r for r, _, _ in refs]
  if not allv:
    return "<i>暂无数据（指标从本 run 起记录）</i>"
  vmin, vmax = min(allv), max(allv)
  pad = max(0.01, (vmax - vmin) * 0.1)
  vmin, vmax = vmin - pad, vmax + pad
  steps = [st for s, _ in series for st, _ in s]
  it0, it1 = min(steps), max(steps)
  span = max(1, it1 - it0)

  def X(st):
    return pl + pw * (st - it0) / span

  def Y(v):
    return pt + ph * (1 - (v - vmin) / (vmax - vmin))

  svg = []
  for val, label, color in refs:
    yy = Y(val)
    svg.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{W - pr}" y2="{yy:.1f}" stroke="{color}" '
               f'stroke-dasharray="5" vector-effect="non-scaling-stroke" opacity="0.8" />')
    svg.append(f'<text x="{pl + 4}" y="{float(yy) - 5:.1f}" font-size="13" fill="{color}">{label}</text>')
  for values, color in series:
    if not values:
      continue
    n = max(2, min(len(values), 400))
    idx = np.linspace(0, len(values) - 1, n).astype(int)
    pts = " ".join(f"{X(values[i][0]):.1f},{Y(values[i][1]):.1f}" for i in idx)
    lx, ly = X(values[-1][0]), Y(values[-1][1])
    svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" '
               f'vector-effect="non-scaling-stroke" points="{pts}" />')
    svg.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3" fill="{color}" />')
    svg.append(f'<text x="{lx - 4:.1f}" y="{ly - 6:.1f}" font-size="12" fill="{color}" '
               f'text-anchor="end">{values[-1][1]:.3f}</text>')
  svg.append(f'<text x="{pl}" y="{H - 6}" font-size="12" fill="#889">iter {it0}</text>')
  svg.append(f'<text x="{W - pr}" y="{H - 6}" font-size="12" fill="#889" '
             f'text-anchor="end">iter {it1}</text>')
  return f'<svg viewBox="0 0 {W} {H}" width="100%">' + "".join(svg) + "</svg>"


def read_status(run: str, total: int) -> dict:
  import numpy as np
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  root = os.path.join(MJLAB_DIR, "logs", "rsl_rl", _EXPERIMENT)
  runs = sorted(glob.glob(os.path.join(root, f"*_{run}")), key=os.path.getmtime)
  out = {"iter": 0, "total": total, "stage": "-", "ckpt": -1, "rew": "-",
         "eplen_last": "-", "chart": "", "tchart": "", "pct": 0, "eta": "-", "elapsed": "-",
         "params": "", "dchart": "<i>暂无数据</i>", "drift_last": "-",
         "hchart": "<i>暂无数据</i>", "h_last": "-",
         "rschart": "<i>暂无数据</i>", "rise_last": "-", "settle_last": "-"}
  if not runs:
    return out
  run_dir = runs[-1]
  out["params"] = read_params_html(run_dir)
  # started time from dir name 2026-10-25_21-16-42
  try:
    t0 = time.mktime(time.strptime(os.path.basename(run_dir)[:19], "%Y-%m-%d_%H-%M-%S"))
    secs = time.time() - t0
    out["elapsed"] = f"{int(secs // 3600)}h{int(secs % 3600 // 60):02d}m"
  except Exception:
    pass
  cks = glob.glob(os.path.join(run_dir, "model_*.pt"))
  if cks:
    out["ckpt"] = max(int(os.path.basename(c)[6:-3]) for c in cks)

  ea = EventAccumulator(run_dir)
  ea.Reload()
  tags = ea.Tags()["scalars"]
  if "Train/mean_episode_length" in tags:
    l = [(s.step, s.value, s.wall_time) for s in ea.Scalars("Train/mean_episode_length")]
    out["iter"] = l[-1][0]
    out["pct"] = round(100 * out["iter"] / total, 1)
    out["eplen_last"] = round(l[-1][1])
    per_iter = (l[-1][2] - l[0][2]) / max(1, l[-1][0] - l[0][0])
    remain = (total - out["iter"]) * per_iter
    out["eta"] = f"{int(remain // 3600)}h{int(remain % 3600 // 60):02d}m"
    # SVG 曲线（viewBox 自适应，随窗口宽度缩放）
    out["chart"] = render_chart(l)
    # 跟踪质量：mjlab 的 Episode_Reward/* 已是「episode 奖励和 ÷ 20 s」
    # = 每秒均值；配合每项 clip±1 与 dt=0.01，数值直接等于"每步裁剪均值"
    # （1.0 = 完美跟踪），不要再次归一化。
    def _rate(tag):
      if tag not in tags:
        return []
      return [(s.step, s.value) for s in ea.Scalars(tag)]

    out["tchart"] = render_multi_chart(
      [
        (_rate("Episode_Reward/track_lin_vel"), "#7c7"),
        (_rate("Episode_Reward/track_yaw_vel"), "#6ac"),
        (_rate("Episode_Reward/height"), "#e8a04c"),
      ],
      ["track_lin", "track_yaw", "height"],
    )
    drift = _rate("Episode_Metrics/stand_drift")
    if drift:
      out["dchart"] = render_multi_chart([(drift, "#e06666")], ["stand_drift"])
      out["drift_last"] = f"{drift[-1][1]:+.3f} m/s"
    h_act = _rate("Episode_Metrics/stand_height")
    h_cmd = _rate("Episode_Metrics/stand_height_cmd")
    if h_act:
      out["hchart"] = render_pos_chart([(h_act, "#7c7"), (h_cmd, "#e8a04c")])
      out["h_last"] = f"{h_act[-1][1]:.3f} m"
    rise = _rate("Episode_Metrics/rise_time")
    settle = _rate("Episode_Metrics/settle_time")
    if rise:
      refs = ((2.0, "起立目标 2s", "#e8a04c"), (0.2, "稳定目标 0.2s", "#6ac"))
      out["rschart"] = render_pos_chart([(rise, "#7c7"), (settle, "#6ac")], refs)
      out["rise_last"] = f"{rise[-1][1]:.2f} s"
      if settle:
        out["settle_last"] = f"{settle[-1][1]:.2f} s"
  if "Train/mean_reward" in tags:
    out["rew"] = round(ea.Scalars("Train/mean_reward")[-1].value)
  if "Curriculum/command_standing/stage" in tags:
    st = ea.Scalars("Curriculum/command_standing/stage")[-1].value
    p = ea.Scalars("Curriculum/command_standing/standing_prob")[-1].value
    names = {1.0: "Stage 0 全站定", 0.5: "Stage 1 机动(50%)"}  # 按 standing_prob
    out["stage"] = f'{names.get(p, p)} (stage={st:.0f})'
  return out


class Handler(BaseHTTPRequestHandler):
  run = "flat_curri2"
  total = 3000

  def do_GET(self):
    s = read_status(self.run, self.total)
    body = PAGE.format(run=html.escape(self.run), **{k: s[k] for k in
        ("iter", "total", "pct", "eta", "elapsed", "stage", "ckpt", "eplen_last", "rew",
         "chart", "tchart", "params", "dchart", "drift_last", "hchart", "h_last",
         "rschart", "rise_last", "settle_last")})
    data = body.encode()
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def log_message(self, *a):  # 静默访问日志
    pass


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", default="flat_curri2")
  ap.add_argument("--total", type=int, default=3000)
  ap.add_argument("--port", type=int, default=8765)
  ap.add_argument("--experiment", default="wheelleg_v12_mjlab")
  args = ap.parse_args()
  global _EXPERIMENT
  _EXPERIMENT = args.experiment
  Handler.run, Handler.total = args.run, args.total
  print(f"dashboard: http://127.0.0.1:{args.port}  (run={args.run})", flush=True)
  HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
  raise SystemExit(main())

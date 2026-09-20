#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CARLA 连通性 / 渲染速度冒烟测试：起一个车 + 相机，跑几帧，存一张图。

用法: python smoke_test.py [rpc_port] [map_name]
"""
import os
import sys
import time

import numpy as np

import carla
import cv2

port = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
mapname = sys.argv[2] if len(sys.argv) > 2 else "Town10HD_Opt"

print(f"[smoke] connect 127.0.0.1:{port}")
client = carla.Client("127.0.0.1", port)
client.set_timeout(90.0)
print("[smoke] available maps:", client.get_available_maps())

t0 = time.time()
world = client.load_world(mapname)
print(f"[smoke] load_world('{mapname}') ok in {time.time() - t0:.1f}s -> "
      f"{world.get_map().name}  spawn_points={len(world.get_map().get_spawn_points())}")

s = world.get_settings()
old = s
s.synchronous_mode = True
s.fixed_delta_seconds = 0.05
world.apply_settings(s)

bps = [b for b in world.get_blueprint_library().filter("vehicle.*")
       if b.has_attribute("number_of_wheels") and int(b.get_attribute("number_of_wheels")) >= 4]
bp = bps[0]
sp = world.get_map().get_spawn_points()[0]
veh = world.spawn_actor(bp, sp)
print(f"[smoke] spawned {veh.type_id} at {sp.location}")

cbp = world.get_blueprint_library().find("sensor.camera.rgb")
cbp.set_attribute("image_size_x", "320")
cbp.set_attribute("image_size_y", "180")
frames = []
cam = world.spawn_actor(cbp, carla.Transform(carla.Location(x=-6.0, z=3.0),
                                             carla.Rotation(pitch=-15.0)), attach_to=veh)
cam.listen(lambda img: frames.append(img))

print("[smoke] ticking ...")
times = []
for i in range(6):
    t = time.time()
    world.tick()
    times.append(time.time() - t)
    print(f"   tick {i}: {times[-1]:.2f}s  images={len(frames)}")

if frames:
    img = frames[-1]
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    bgr = arr[:, :, :3]
    cv2.imwrite("/tmp/smoke.png", bgr)
    print(f"[smoke] saved /tmp/smoke.png  mean={bgr.mean():.1f}  "
          f"(全黑=渲染没出来, mean 应该 >5)")
else:
    print("[smoke] !! 没收到任何相机图像")

cam.destroy()
veh.destroy()
s.synchronous_mode = False
world.apply_settings(old)
print("[smoke] OK，平均每 tick %.2fs" % (sum(times) / len(times)))

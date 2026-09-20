import argparse
import os
import random
from pathlib import Path
import json
import math
import re

import lpips
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from dggt.models.vggt import VGGT
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_split_gs
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from gsplat.rendering import rasterization

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
META_EXTS = {".mrk", ".nav", ".obs", ".rtk"}
EARTH_RADIUS_M = 6378137.0


def compute_lifespan_loss(gamma):
    return torch.mean(torch.abs(1 / (gamma + 1e-6)))


def alpha_t(t, t0, alpha, gamma0=1, gamma1=0.1):
    sigma = torch.log(torch.tensor(gamma1, device=gamma0.device)) / ((gamma0) ** 2 + 1e-6)
    return (alpha * torch.exp(sigma * (t0 - t) ** 2)).float()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/root/autodl-fs/dggt-main/mydata")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="logs/mydata_exp")
    parser.add_argument("--sequence_length", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=378)
    parser.add_argument("--max_epoch", type=int, default=50000)
    parser.add_argument("--save_image", type=int, default=100)
    parser.add_argument("--save_ckpt", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--scene_names", type=str, nargs="*", default=None)
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def _parse_mrk_file(mrk_path):
    records = []
    with open(mrk_path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 6:
                parts = re.split(r'\s+', line)
            if len(parts) < 6:
                continue
            try:
                rel_time = float(parts[1])
            except Exception:
                continue
            lat = lon = ellh = None
            for seg in parts:
                if 'Lat' in seg:
                    try:
                        lat = float(seg.split(',')[0])
                    except Exception:
                        pass
                elif 'Lon' in seg:
                    try:
                        lon = float(seg.split(',')[0])
                    except Exception:
                        pass
                elif 'Ellh' in seg:
                    try:
                        ellh = float(seg.split(',')[0])
                    except Exception:
                        pass
            if lat is None or lon is None:
                m = re.search(r'([\-\d\.]+),Lat\s*([\-\d\.]+),Lon\s*([\-\d\.]+),Ellh', line)
                if m:
                    lat, lon, ellh = map(float, m.groups())
            if lat is not None and lon is not None and ellh is not None:
                records.append({'time': rel_time, 'lat': lat, 'lon': lon, 'ellh': ellh, 'raw': line})
    records.sort(key=lambda x: x['time'])
    return records


def _latlon_to_local_xy(lat, lon, ref_lat, ref_lon):
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    x = EARTH_RADIUS_M * dlon * math.cos(math.radians(ref_lat))
    y = EARTH_RADIUS_M * dlat
    return x, y


def _build_pose_matrix(x, y, z):
    pose = torch.eye(4, dtype=torch.float32)
    pose[0, 3] = float(x)
    pose[1, 3] = float(y)
    pose[2, 3] = float(z)
    return pose


def _write_intrinsics(out_path, width, height, fx=None, fy=None, cx=None, cy=None):
    if fx is None:
        fx = fy = float(max(width, height)) * 1.2
    if fy is None:
        fy = fx
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0
    K = [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]]
    with open(out_path, 'w') as f:
        for row in K:
            f.write(' '.join(f'{v:.8f}' for v in row) + '\n')
    return K


def prepare_scene_exports(scene_dir, min_gap=1):
    scene_dir = Path(scene_dir)
    image_files = sorted([p for p in scene_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not image_files:
        return None
    mrk_files = sorted(scene_dir.glob('*.MRK')) + sorted(scene_dir.glob('*.mrk'))
    if not mrk_files:
        return None
    records = _parse_mrk_file(mrk_files[0])
    if not records:
        return None
    ref_lat, ref_lon, ref_ellh = records[0]['lat'], records[0]['lon'], records[0]['ellh']
    ego_pose_dir = scene_dir / 'ego_pose'
    intr_dir = scene_dir / 'intrinsics'
    ego_pose_dir.mkdir(exist_ok=True)
    intr_dir.mkdir(exist_ok=True)
    first_img = Image.open(image_files[0])
    width, height = first_img.size
    K = _write_intrinsics(intr_dir / '0.txt', width, height)
    img_to_rec = []
    if len(records) == len(image_files):
        img_to_rec = list(zip(image_files, records))
    else:
        for i, img in enumerate(image_files):
            rec = records[min(i * max(min_gap, 1), len(records) - 1)]
            img_to_rec.append((img, rec))
    for idx, (_, rec) in enumerate(img_to_rec):
        x, y = _latlon_to_local_xy(rec['lat'], rec['lon'], ref_lat, ref_lon)
        z = rec['ellh'] - ref_ellh
        pose = _build_pose_matrix(x, y, z)
        payload = {
            'frame_id': idx,
            'timestamp': rec['time'],
            'lat': rec['lat'],
            'lon': rec['lon'],
            'ellh': rec['ellh'],
            'camera': {'width': width, 'height': height},
            'camera_intrinsics': K,
            'camera_extrinsics_world': pose.tolist(),
            'ego_pose_world': [float(x), float(y), float(z)],
            'reference_gps': {'lat': ref_lat, 'lon': ref_lon, 'ellh': ref_ellh},
            'source_mrk': mrk_files[0].name,
        }
        with open(ego_pose_dir / f'frame_{idx:04d}_ego.json', 'w') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return {'scene_dir': str(scene_dir), 'frames': len(img_to_rec), 'reference_gps': {'lat': ref_lat, 'lon': ref_lon, 'ellh': ref_ellh}}


class DroneSceneDataset(Dataset):
    def __init__(self, root_dir, scene_names=None, sequence_length=4, auto_prepare=True, image_size=384):
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        if scene_names is None:
            scene_names = [p.name for p in self.root_dir.iterdir() if p.is_dir()]
        self.scenes = []
        self.prepared = []
        for scene_name in sorted(scene_names):
            scene_dir = self.root_dir / scene_name
            if not scene_dir.is_dir():
                continue
            if auto_prepare and not (scene_dir / 'ego_pose').exists():
                prepare_scene_exports(scene_dir)
            images = sorted([p for p in scene_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
            if len(images) >= sequence_length:
                self.scenes.append((scene_name, images))
        if not self.scenes:
            raise RuntimeError(f'未找到可训练场景：{root_dir}')

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name, image_files = self.scenes[idx]
        start = random.randint(0, len(image_files) - self.sequence_length)
        seq = image_files[start:start + self.sequence_length]
        imgs = []
        for p in seq:
            img = Image.open(p)
            if img.mode == 'RGBA':
                bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img)
            img = img.convert('RGB')
            w, h = img.size
            new_w = self.image_size
            new_h = max(14, round(h * (new_w / w) / 14) * 14)
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            img = T.ToTensor()(img)
            if img.shape[1] > self.image_size:
                img = img[:, (img.shape[1] - self.image_size) // 2:(img.shape[1] - self.image_size) // 2 + self.image_size, :]
            if img.shape[2] < self.image_size:
                img = torch.nn.functional.pad(img, (0, self.image_size - img.shape[2], 0, max(0, self.image_size - img.shape[1])), value=1.0)
            imgs.append(img[:, :self.image_size, :self.image_size])

        images = torch.stack(imgs, dim=0)
        masks = torch.zeros_like(images)
        timestamps = torch.linspace(0, self.sequence_length / 4, self.sequence_length)
        meta_files = []
        for ext in META_EXTS:
            meta_files.extend(sorted((self.root_dir / scene_name).glob(f'*{ext}')))
        return {'images': images, 'masks': masks, 'timestamps': timestamps, 'image_paths': [str(p) for p in seq], 'meta_files': [str(p) for p in meta_files]}


def main(args):
    if args.prepare_only:
        print(f"数据目录：{args.image_dir}")
        scene_names = args.scene_names
        if scene_names is None:
            scene_names = [p.name for p in Path(args.image_dir).iterdir() if p.is_dir()]
        for scene_name in scene_names:
            scene_dir = Path(args.image_dir) / scene_name
            if not scene_dir.is_dir():
                print(f"[跳过] 场景不存在: {scene_dir}")
                continue
            result = prepare_scene_exports(scene_dir)
            if result is None:
                print(f"[跳过] 无法从 {scene_dir} 生成 ego_pose/intrinsics（可能没有 MRK 或图像）")
            else:
                print(f"[完成] {scene_name}: {result['frames']} 帧, 参考坐标 {result['reference_gps']}")
        print("已完成预处理导出。")
        return

    use_ddp = torch.cuda.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aligned_image_size = max(14, (args.image_size // 14) * 14)
    if aligned_image_size != args.image_size:
        print(f"[提示] image_size={args.image_size} 不是 14 的倍数，已自动调整为 {aligned_image_size}")
    dataset = DroneSceneDataset(args.image_dir, scene_names=args.scene_names, sequence_length=args.sequence_length, auto_prepare=True, image_size=aligned_image_size)
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=True)

    if (not use_ddp) or args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "ckpt"), exist_ok=True)

    model = VGGT().to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.train()
    train_model = model
    if use_ddp:
        model = DDP(model, device_ids=[args.local_rank])
        model._set_static_graph()
        train_model = model.module

    lpips_loss_fn = lpips.LPIPS(net="alex").to(device)
    binary_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    for p in train_model.parameters():
        p.requires_grad = False
    for head_name in ["gs_head", "instance_head", "sky_model"]:
        for p in getattr(train_model, head_name).parameters():
            p.requires_grad = True

    optimizer = AdamW([
        {"params": train_model.gs_head.parameters(), "lr": 4e-5},
        {"params": train_model.instance_head.parameters(), "lr": 4e-5},
        {"params": train_model.sky_model.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

    warmup_iterations = 1000
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / warmup_iterations, 1.0) * 0.5 * (1 + torch.cos(torch.tensor(torch.pi * step / args.max_epoch))))

    for step in tqdm(range(args.max_epoch)):
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in dataloader:
            images = batch["images"].to(device)
            sky_mask = batch["masks"].to(device).permute(0, 1, 3, 4, 2)
            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch["timestamps"][0].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                predictions = model(images)
                H, W = images.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]

                depth_map = predictions["depth"][0]
                point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None, ...]
                point_map = torch.from_numpy(point_map).to(device).float()
                gs_map = predictions["gs_map"]
                gs_conf = predictions["gs_conf"]
                dy_map = predictions["dynamic_conf"].squeeze(-1)

                static_mask = torch.ones_like(bg_mask)
                static_points = point_map[static_mask].reshape(-1, 3)
                static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
                static_opacity = static_opacity * (1 - dy_map[static_mask].sigmoid())
                static_gs_conf = gs_conf[static_mask]
                frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
                gs_timestamps = timestamps[frame_idx]

                dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
                for i in range(dy_map.shape[1]):
                    point_map_i = point_map[:, i]
                    bg_mask_i = bg_mask[:, i]
                    dynamic_points.append(point_map_i[bg_mask_i].reshape(-1, 3))
                    drgb, dop, ds, drot = get_split_gs(gs_map[:, i], bg_mask_i)
                    dynamic_rgbs.append(drgb)
                    dynamic_opacitys.append(dop * dy_map[:, i][bg_mask_i].sigmoid())
                    dynamic_scales.append(ds)
                    dynamic_rotations.append(drot)

                chunked_renders, chunked_alphas = [], []
                for idx in range(extrinsic.shape[0]):
                    t0 = timestamps[idx]
                    static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0=static_gs_conf)
                    world_points, rgbs, opacity, scales, rotation = concat_list(
                        [static_points, static_rgbs, static_opacity_, static_scales, static_rotations],
                        [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]],
                    )
                    renders_chunk, alphas_chunk, _ = rasterization(
                        means=world_points, quats=rotation, scales=scales, opacities=opacity, colors=rgbs,
                        viewmats=extrinsic[idx][None], Ks=intrinsic[idx][None], width=W, height=H,
                    )
                    chunked_renders.append(renders_chunk)
                    chunked_alphas.append(alphas_chunk)

                renders = torch.cat(chunked_renders, dim=0)
                alphas = torch.cat(chunked_alphas, dim=0)
                bg_render = train_model.sky_model(images, extrinsic, intrinsic)
                renders = alphas * renders + (1 - alphas) * bg_render
                rendered_image = renders.permute(0, 3, 1, 2)
                target_image = images[0]

                loss = F.l1_loss(rendered_image, target_image)
                loss = loss + F.l1_loss(alphas, 1 - sky_mask[0, ..., 0][..., None])
                loss = loss + 0.01 * compute_lifespan_loss(static_gs_conf)
                loss = loss + 0.05 * binary_loss_fn(dy_map[0], torch.zeros_like(dy_map[0]))
                loss = loss + 0.05 * lpips_loss_fn(rendered_image, target_image).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

        if (not use_ddp) or args.local_rank == 0:
            print(f"[{step}/{args.max_epoch}] Loss: {loss.item():.4f}")
            if step % args.save_image == 0:
                rnd = random.randint(0, rendered_image.shape[0] - 1)
                rendered = rendered_image[rnd].detach().cpu().clamp(0, 1)
                target = target_image[rnd].detach().cpu().clamp(0, 1)
                dy_map_rgb = torch.sigmoid(dy_map[0, rnd]).detach().cpu().unsqueeze(0).repeat(3, 1, 1)
                sem_rgb = alphas[rnd, ..., 0].unsqueeze(0).repeat(3, 1, 1).cpu()
                combined = torch.cat([target, rendered, dy_map_rgb, sem_rgb], dim=-1)
                T.ToPILImage()(combined).save(os.path.join(args.log_dir, "images", f"step_{step}_frame_{rnd}.png"))
            if step > 0 and step % args.save_ckpt == 0:
                ckpt_path = os.path.join(args.log_dir, "ckpt", "model_latest.pt")
                torch.save(train_model.state_dict(), ckpt_path)
                print(f"[Checkpoint] Saved model at step {step} to {ckpt_path}")


if __name__ == "__main__":
    main(parse_args())

"""DGGT 后端 -> SAM 3D 微服务的 HTTP 客户端（运行在 dggt 环境，无重依赖）。

SAM 3D 微服务运行在独立的 sam3d-objects 环境（端口 8001），本模块仅负责
通过 HTTP 转发图片/掩码并取回重建结果，避免把 torch/gsplat 等冲突依赖
引入 dggt 后端进程。
"""
import os
from typing import Optional, Tuple

import httpx

# 微服务地址：可通过环境变量覆盖
SAM3D_SERVICE_URL = os.environ.get("SAM3D_SERVICE_URL", "http://127.0.0.1:8001")

# 重建结果在本机的保存目录（供 /api/sam3d/import 直接读取 .ply）
SAM3D_PLY_DIR = os.environ.get("SAM3D_PLY_DIR", "/root/autodl-tmp/sam3d_outputs")


def _ensure_ply_dir() -> str:
    os.makedirs(SAM3D_PLY_DIR, exist_ok=True)
    return SAM3D_PLY_DIR


class SAM3DClient:
    """SAM 3D 微服务客户端。"""

    def __init__(self, base_url: str = SAM3D_SERVICE_URL):
        self.base_url = base_url.rstrip("/")

    async def health(self) -> dict:
        """查询微服务健康状态。"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    async def wait_ready(self, timeout: float = 300.0, interval: float = 2.0) -> bool:
        """等待微服务可用。

        `/unload` 会让微服务进程退出并由 supervisor 重启（用于释放显存），
        因此**连续两次重建**之间必须等服务重新起来，否则会 "Server disconnected"。
        """
        import asyncio
        import time as _time
        deadline = _time.time() + timeout
        last_err = None
        while _time.time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.get(f"{self.base_url}/health")
                    if resp.status_code == 200:
                        return True
            except Exception as e:  # noqa: BLE001
                last_err = e
            await asyncio.sleep(interval)
        raise RuntimeError(f"SAM 3D 服务在 {timeout:.0f}s 内未就绪: {last_err}")

    async def reconstruct(
        self,
        image_bytes: bytes,
        mask_bytes: Optional[bytes] = None,
        seed: Optional[int] = None,
        format: str = "ply",
        image_filename: str = "image.png",
        mask_filename: str = "mask.png",
    ) -> dict:
        """调用微服务重建 3D 对象，返回 {ply_path, glb_path, pose, num_points, seed}。

        微服务内部会生成 ply/glb 文件；这里通过 /outputs/{filename} 把 .ply
        下载到本机 SAM3D_PLY_DIR，供后续导入 DGGT 场景时本地读取。
        """
        import asyncio
        files = {"image": (image_filename, image_bytes, "image/png")}
        if mask_bytes is not None:
            files["mask"] = (mask_filename, mask_bytes, "image/png")
        data = {"format": format}
        if seed is not None:
            data["seed"] = str(seed)

        result = None
        last_err = None
        for attempt in range(3):
            # 上一次重建结束会 unload → supervisor 重启进程，先等服务就绪
            await self.wait_ready()
            try:
                async with httpx.AsyncClient(timeout=900.0) as client:
                    resp = await client.post(
                        f"{self.base_url}/reconstruct", files=files, data=data
                    )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"SAM 3D 服务重建失败 (HTTP {resp.status_code}): {resp.text[:500]}"
                    )
                result = resp.json()
                break
            except (httpx.TransportError, httpx.HTTPError) as e:
                last_err = e
                if attempt >= 2:
                    raise RuntimeError(f"连接 SAM 3D 服务失败: {e}")
                await asyncio.sleep(4)
        if result is None:
            raise RuntimeError(f"SAM 3D 服务重建失败: {last_err}")

        # 把 .ply 落到本机可读路径（微服务与后端通常同机，直接复制即可；
        # 跨机部署时则通过 /outputs/{filename} 走 HTTP 下载）。
        remote_ply = result.get("ply_path")
        if remote_ply:
            local_ply = self._download_output(remote_ply)
            result["local_ply_path"] = local_ply
        return result

    def _download_output(self, remote_path: str) -> str:
        filename = os.path.basename(remote_path)
        local_path = os.path.join(_ensure_ply_dir(), filename)
        if os.path.exists(remote_path):
            if os.path.abspath(remote_path) != os.path.abspath(local_path):
                import shutil
                shutil.copyfile(remote_path, local_path)
            return local_path
        # 微服务与后端不在同一台机器时，走 HTTP 下载
        with httpx.Client(timeout=300.0) as client:
            resp = client.get(f"{self.base_url}/outputs/{filename}")
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
        return local_path

    async def unload(self) -> dict:
        """请求微服务卸载已加载的模型，释放显存。"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.base_url}/unload")
            resp.raise_for_status()
            return resp.json()


# 便捷单例
_client: Optional[SAM3DClient] = None


def get_sam3d_client() -> SAM3DClient:
    global _client
    if _client is None:
        _client = SAM3DClient()
    return _client

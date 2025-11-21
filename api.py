"""
FastAPI 推理服务
使用 vLLM 进行 Qwen2.5-VL 图片分类推理
支持上传文件和OSS URL两种方式
"""

import io
import json
import gc
import os
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

import oss2
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from vllm import LLM, SamplingParams

# ============ 全局变量 ============
llm_engine = None
sampling_params = None
oss_bucket = None  # OSS Bucket实例


# ============ 配置 ============
class Config:
    MODEL_PATH = "/data/hx/LLaMA-Factory/output/qwen2_5vl_lora_classify"
    MAX_TOKENS = 32
    TEMPERATURE = 0.0
    GPU_MEMORY_UTILIZATION = 0.9

    # 阿里云OSS配置（从环境变量读取）
    OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID", "")
    OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET", "")
    # 使用内网endpoint，不走公网，节省流量费用并提高速度
    # 内网: oss-cn-beijing-internal.aliyuncs.com (同区域ECS访问)
    # 外网: oss-cn-beijing.aliyuncs.com (公网访问)
    OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "oss-cn-beijing-internal.aliyuncs.com")
    OSS_BUCKET_NAME = os.getenv("OSS_BUCKET_NAME", "ts-bigdata-chart-prd")

    # 推理提示词
    PROMPT_TEMPLATE = "请判断这张商品图片的角度类别。类别包括：全身模特、其他角度、口袋特写、商标特写、正面平铺、正面模特、背面平铺、背面模特、腰部特写、裤脚特写。请直接回答类别名称。"


# ============ 数据模型 ============
class PredictResponse(BaseModel):
    """推理响应"""
    image_name: str
    category: str
    confidence: Optional[float] = None


class PredictByOssRequest(BaseModel):
    """通过OSS路径推理的请求"""
    object_key: str  # OSS对象路径，如: products/image001.jpg
    bucket_name: Optional[str] = None  # 可选，默认使用配置的bucket
    image_name: Optional[str] = None


class BatchPredictByOssRequest(BaseModel):
    """批量OSS路径推理的请求"""
    object_keys: List[str]
    bucket_name: Optional[str] = None


class BatchPredictResponse(BaseModel):
    """批量推理响应"""
    results: List[PredictResponse]
    total: int
    errors: Optional[List[dict]] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    oss_connected: bool


# ============ 辅助函数 ============
def prepare_prompt() -> str:
    """准备推理prompt（Qwen2.5-VL官方格式）"""
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{Config.PROMPT_TEMPLATE}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def download_image_from_oss(object_key: str, bucket_name: Optional[str] = None) -> Image.Image:
    """
    从阿里云OSS下载图片

    Args:
        object_key: OSS对象路径，如 'products/image001.jpg'
        bucket_name: bucket名称，默认使用配置的bucket

    Returns:
        PIL.Image对象

    Raises:
        HTTPException: 下载失败或图片无效
    """
    global oss_bucket

    try:
        # 使用指定bucket或默认bucket
        if bucket_name and bucket_name != Config.OSS_BUCKET_NAME:
            # 创建临时bucket对象
            auth = oss2.Auth(Config.OSS_ACCESS_KEY_ID, Config.OSS_ACCESS_KEY_SECRET)
            temp_bucket = oss2.Bucket(auth, Config.OSS_ENDPOINT, bucket_name)
            bucket = temp_bucket
        else:
            bucket = oss_bucket

        # 检查文件是否存在
        if not bucket.object_exists(object_key):
            raise HTTPException(
                status_code=404,
                detail=f"OSS文件不存在: {object_key}"
            )

        # 下载文件
        result = bucket.get_object(object_key)
        image_data = io.BytesIO(result.read())

        # 转换为PIL Image
        image = Image.open(image_data)

        # 验证图片
        image.verify()

        # 重新加载（verify后需要重新打开）
        image_data.seek(0)
        image = Image.open(image_data)

        return image

    except oss2.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"OSS文件不存在: {object_key}")
    except oss2.exceptions.NoSuchBucket:
        raise HTTPException(status_code=404, detail=f"OSS Bucket不存在: {bucket_name}")
    except oss2.exceptions.AccessDenied:
        raise HTTPException(status_code=403, detail="OSS访问被拒绝，请检查AccessKey权限")
    except oss2.exceptions.ServerError as e:
        raise HTTPException(status_code=500, detail=f"OSS服务器错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OSS图片处理失败: {str(e)}")


def predict_image(image: Image.Image, prompt: str) -> str:
    """对单张图片进行推理"""
    global llm_engine, sampling_params

    # 确保图片是RGB模式
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 使用vLLM进行推理
    outputs = llm_engine.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling_params
    )

    # 提取响应
    response = outputs[0].outputs[0].text.strip()
    return response


# ============ 生命周期管理 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时加载模型，关闭时清理"""
    global llm_engine, sampling_params, oss_bucket

    print("=" * 60)
    print("🚀 启动 FastAPI 推理服务（OSS专用版）...")
    print("=" * 60)

    # 初始化OSS客户端
    try:
        auth = oss2.Auth(Config.OSS_ACCESS_KEY_ID, Config.OSS_ACCESS_KEY_SECRET)
        oss_bucket = oss2.Bucket(auth, Config.OSS_ENDPOINT, Config.OSS_BUCKET_NAME)
        # 测试连接
        oss_bucket.get_bucket_info()
        print(f"✓ OSS连接成功: {Config.OSS_BUCKET_NAME}")
    except Exception as e:
        print(f"❌ OSS初始化失败: {str(e)}")
        print("  请检查OSS配置")
        oss_bucket = None

    # 初始化 vLLM
    print(f"\n⏳ 正在加载模型: {Config.MODEL_PATH}")
    llm_engine = LLM(
        model=Config.MODEL_PATH,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=Config.GPU_MEMORY_UTILIZATION,
    )
    print("✓ 模型加载完成")

    # 创建采样参数
    sampling_params = SamplingParams(
        max_tokens=Config.MAX_TOKENS,
        temperature=Config.TEMPERATURE,
        top_p=1.0,
    )
    print(f"✓ 采样参数配置完成 (max_tokens={Config.MAX_TOKENS}, temperature={Config.TEMPERATURE})")

    print("\n" + "=" * 60)
    print("✅ 服务启动成功！")
    print("=" * 60)
    print(f"📍 API文档: http://0.0.0.0:8888/docs")
    print(f"📍 健康检查: http://0.0.0.0:8888/health")
    print("=" * 60 + "\n")

    yield

    # 关闭时清理
    print("\n🛑 正在关闭服务...")
    llm_engine = None
    print("✓ 服务已关闭")


# ============ FastAPI 应用 ============
app = FastAPI(
    title="Qwen2.5-VL 图片分类 API",
    description="基于 vLLM 的商品图片角度分类推理服务",
    version="1.0.0",
    lifespan=lifespan
)


# ============ API 路由 ============
@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "message": "Qwen2.5-VL 图片分类 API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if llm_engine is not None else "loading",
        model_loaded=llm_engine is not None,
        oss_connected=oss_bucket is not None
    )


@app.post("/predict/oss", response_model=PredictResponse, tags=["Inference"])
async def predict_by_oss(request: PredictByOssRequest):
    """
    单张图片推理（OSS路径方式，推荐）

    - **object_key**: OSS对象路径，如 'products/image001.jpg'
    - **bucket_name**: 可选，bucket名称（默认使用配置的bucket）
    - **image_name**: 可选，图片名称（用于返回结果标识）

    返回预测的类别

    优势：
    - 无需生成签名URL
    - 支持私有bucket
    - 更安全高效

    示例:
    ```json
    {
        "object_key": "products/image001.jpg",
        "image_name": "产品001"
    }
    ```
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    if oss_bucket is None:
        raise HTTPException(status_code=503, detail="OSS未初始化，请检查配置")

    pil_image = None
    try:
        # 从OSS下载图片
        pil_image = download_image_from_oss(
            request.object_key,
            request.bucket_name
        )

        # 准备prompt
        prompt = prepare_prompt()

        # 推理
        result = predict_image(pil_image, prompt)

        # 确定图片名称
        image_name = request.image_name or request.object_key.split('/')[-1]

        return PredictResponse(
            image_name=image_name,
            category=result
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")
    finally:
        # 显式清理图片内存
        if pil_image is not None:
            try:
                pil_image.close()
            except:
                pass
            del pil_image
            # 强制垃圾回收
            gc.collect()


@app.post("/predict/batch/oss", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch_by_oss(request: BatchPredictByOssRequest):
    """
    批量图片推理（OSS路径方式，推荐）

    - **object_keys**: OSS对象路径列表
    - **bucket_name**: 可选，bucket名称（默认使用配置的bucket）

    返回所有图片的预测结果，包含错误信息

    示例:
    ```json
    {
        "object_keys": [
            "products/image001.jpg",
            "products/image002.jpg",
            "products/image003.jpg"
        ]
    }
    ```
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    if oss_bucket is None:
        raise HTTPException(status_code=503, detail="OSS未初始化，请检查配置")

    if len(request.object_keys) == 0:
        raise HTTPException(status_code=400, detail="请至少提供一个OSS对象路径")

    if len(request.object_keys) > 100:
        raise HTTPException(status_code=400, detail="单次最多支持100张图片")

    try:
        results = []
        errors = []
        prompt = prepare_prompt()

        # 逐张处理，立即清理内存
        for idx, object_key in enumerate(request.object_keys):
            pil_image = None
            try:
                # 从OSS下载图片
                pil_image = download_image_from_oss(
                    object_key,
                    request.bucket_name
                )

                # 确保图片是RGB模式
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                # 单张推理
                result = predict_image(pil_image, prompt)

                # 提取图片名称
                image_name = object_key.split('/')[-1]

                results.append(PredictResponse(
                    image_name=image_name,
                    category=result
                ))

            except HTTPException as e:
                # 记录错误，继续处理其他图片
                errors.append({
                    "object_key": object_key,
                    "error": e.detail
                })
                continue
            except Exception as e:
                errors.append({
                    "object_key": object_key,
                    "error": str(e)
                })
                continue
            finally:
                # 立即清理当前图片内存
                if pil_image is not None:
                    try:
                        pil_image.close()
                    except:
                        pass
                    del pil_image
                    # 每10张图片强制垃圾回收一次
                    if (idx + 1) % 10 == 0:
                        gc.collect()

        # 最终垃圾回收
        gc.collect()

        return BatchPredictResponse(
            results=results,
            total=len(results),
            errors=errors if errors else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量推理失败: {str(e)}")


# ============ 错误处理 ============
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "接口不存在，请查看 /docs 获取API文档"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8888,  # 修改为8888端口（8000已被占用）
        reload=False,  # 生产环境关闭热重载
        log_level="info"
    )

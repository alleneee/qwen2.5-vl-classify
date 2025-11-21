"""
Qwen2.5-VL vLLM 推理脚本
使用 vLLM 进行高性能推理
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from PIL import Image
from vllm import LLM, SamplingParams


def load_categories(categories_file: Optional[str] = None) -> List[str]:
    """加载类别列表"""
    # 如果指定了类别文件，从文件加载
    if categories_file and Path(categories_file).exists():
        with open(categories_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # 默认从 outputs/categories.json 加载
    default_path = Path("outputs/categories.json")
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 默认类别（不进行映射）
    return [
        "back_model", "back_product", "front_model", "front_product",
        "full_body_model", "leg_cuff", "logo", "other", "pocket", "waist"
    ]


def create_llm(model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
    """创建 vLLM 实例（Qwen2.5-VL官方配置）

    Args:
        model_path: 模型路径
        tensor_parallel_size: 张量并行大小（3B模型建议用1）
        gpu_memory_utilization: GPU 内存利用率
    """
    print(f"初始化 vLLM，模型路径: {model_path}")
    print(f"GPU配置: tensor_parallel_size={tensor_parallel_size}, gpu_memory={gpu_memory_utilization}")

    # 参考vLLM官方Qwen2.5-VL配置
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        max_num_seqs=5,  # 官方推荐值
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    print(f"✓ vLLM初始化完成，使用 {tensor_parallel_size} 张GPU")

    return llm


def create_sampling_params(
        max_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0
) -> SamplingParams:
    """创建采样参数"""
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=[]
    )


def prepare_prompt(categories: List[str], prompt_template: str = None) -> str:
    """准备提示词（Qwen2.5-VL官方格式）"""
    if prompt_template is None:
        # Qwen2.5-VL官方格式：使用<|image_pad|>和完整对话模板
        question = "请判断这张商品图片的角度类别。类别包括：{categories}。请直接回答类别名称。"
        question = question.format(categories="、".join(categories))

        # 官方prompt格式
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt
    else:
        return prompt_template.format(categories="、".join(categories))


def predict_single(
        llm: LLM,
        image_path: str,
        categories: List[str],
        sampling_params: SamplingParams,
        prompt_template: str = None
) -> str:
    """对单张图片进行预测"""

    prompt = prepare_prompt(categories, prompt_template)

    # 加载图片
    image = Image.open(image_path).convert("RGB")

    # 使用vLLM官方格式：llm.generate + multi_modal_data
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling_params
    )

    # 提取响应
    response = outputs[0].outputs[0].text.strip()

    return response


def predict_batch(
        llm: LLM,
        image_paths: List[str],
        categories: List[str],
        sampling_params: SamplingParams,
        batch_size: int = 32,
        prompt_template: str = None
) -> List[dict]:
    """批量预测

    Args:
        llm: vLLM 实例
        image_paths: 图片路径列表
        categories: 类别列表
        sampling_params: 采样参数
        batch_size: 批处理大小
        prompt_template: 提示词模板
    """

    results = []
    total = len(image_paths)

    prompt = prepare_prompt(categories, prompt_template)

    # 分批处理
    for i in range(0, total, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_inputs = []

        print(f"处理批次: {i + 1}-{min(i + batch_size, total)}/{total}")

        # 准备批次输入（使用vLLM官方格式）
        for path in batch_paths:
            image = Image.open(path).convert("RGB")
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            })

        # 批量生成
        outputs = llm.generate(
            batch_inputs,
            sampling_params=sampling_params,
            use_tqdm=False
        )

        # 处理结果
        for path, output in zip(batch_paths, outputs):
            response = output.outputs[0].text.strip()

            results.append({
                "image_path": path,
                "response": response
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="商品图片角度分类推理（vLLM版本）")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="合并后的模型路径"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="单张图片路径"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="图片目录路径（批量推理）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出结果文件路径（JSON）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,  # 单卡A10适中batch
        help="批处理大小（单A10建议32-64）"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,  # Qwen2.5-VL-3B单卡即可，无需多GPU
        help="张量并行大小（仅适用于72B等大模型，3B模型用1即可）"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU内存利用率 (0-1)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度（0表示贪婪解码）"
    )
    parser.add_argument(
        "--categories_file",
        type=str,
        default=None,
        help="类别文件路径（JSON格式）"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="自定义提示词模板，使用{categories}作为占位符"
    )

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("请指定 --image 或 --image_dir")

    # 加载类别
    categories = load_categories(args.categories_file)
    print(f"类别: {categories}\n")

    # 创建 vLLM 实例
    llm = create_llm(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    # 创建采样参数
    sampling_params = create_sampling_params(
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    if args.image:
        # 单张图片推理
        print(f"\n推理图片: {args.image}")
        result = predict_single(llm, args.image, categories, sampling_params, args.prompt_template)
        print(f"模型输出: {result}\n")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({
                    "image_path": args.image,
                    "response": result
                }, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {args.output}")

    elif args.image_dir:
        # 批量推理
        image_dir = Path(args.image_dir)
        image_paths = []

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG"]:
            image_paths.extend(image_dir.glob(ext))

        image_paths = [str(p) for p in sorted(image_paths)]
        print(f"\n找到 {len(image_paths)} 张图片\n")

        results = predict_batch(
            llm,
            image_paths,
            categories,
            sampling_params,
            batch_size=args.batch_size,
            prompt_template=args.prompt_template
        )

        # 显示部分结果
        print("\n前10条推理结果:")
        for r in results[:10]:
            print(f"  {Path(r['image_path']).name}: {r['response']}")

        if len(results) > 10:
            print(f"  ... 共 {len(results)} 条结果")

        # 保存结果
        output_file = args.output or "predictions_vllm.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n完整结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

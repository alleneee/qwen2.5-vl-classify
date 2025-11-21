# Qwen2.5-VL 图片分类项目

基于 [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 进行 Qwen2.5-VL 模型 LoRA 微调，实现商品图片角度分类任务。

## 项目简介

本项目使用 Qwen2.5-VL-3B 多模态大模型，通过 LoRA 微调实现商品图片的角度分类。支持以下类别：

- 全身模特 (full_body_model)
- 正面模特 (front_model)
- 背面模特 (back_model)
- 正面平铺 (front_product)
- 背面平铺 (back_product)
- 口袋特写 (pocket)
- 商标特写 (logo)
- 腰部特写 (waist)
- 裤脚特写 (leg_cuff)
- 其他角度 (other)

## 项目结构

```
.
├── api.py                          # FastAPI 推理服务（OSS版本）
├── inference.py                    # vLLM 批量推理脚本
├── requirements.txt                # Python 依赖
├── .env.example                    # 环境变量配置示例
├── llamafactory/                   # LLaMA Factory 配置文件
│   ├── train.yaml                  # 训练配置
│   ├── inference.yaml              # 推理配置
│   ├── merge_lora.yaml             # LoRA 合并配置
│   ├── dataset_info.json           # 数据集信息
│   └── product_angle_train.json    # 训练数据集
└── data/                           # 数据集目录（被 .gitignore 忽略）
    └── train/                      # 训练图片
```

## 环境要求

- Python 3.10+
- CUDA 11.8+ (用于 GPU 推理)
- 显存：至少 16GB (推荐 A10 或更好的 GPU)

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/alleneee/qwen2.5-vl-classify.git
cd qwen2.5-vl-classify
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 LLaMA Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 4. 配置环境变量（可选，用于 OSS 推理）

复制环境变量示例文件并填写您的配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入您的阿里云 OSS 凭证：

```bash
OSS_ACCESS_KEY_ID=your_access_key_id
OSS_ACCESS_KEY_SECRET=your_access_key_secret
OSS_ENDPOINT=oss-cn-beijing-internal.aliyuncs.com
OSS_BUCKET_NAME=your_bucket_name
```

## 使用方法

### 1. 数据准备

准备训练数据集，格式参考 `llamafactory/product_angle_train.json`：

```json
[
  {
    "messages": [
      {
        "content": "<image>请判断这张商品图片的角度类别。类别包括：...",
        "role": "user"
      },
      {
        "content": "正面模特",
        "role": "assistant"
      }
    ],
    "images": ["data/train/image001.jpg"]
  }
]
```

### 2. LoRA 微调

使用 LLaMA Factory 进行 LoRA 微调：

```bash
cd LLaMA-Factory
llamafactory-cli train ../llamafactory/train.yaml
```

**训练配置说明**（`llamafactory/train.yaml`）：
- **LoRA 参数**: rank=32, alpha=32, dropout=0.1
- **学习率**: 5e-5
- **训练轮数**: 12 epochs
- **批次大小**: 2 (per device)
- **梯度累积**: 2 步
- **优化器**: AdamW + Cosine scheduler

### 3. 合并 LoRA 权重

训练完成后，合并 LoRA 权重到基础模型：

```bash
llamafactory-cli export ../llamafactory/merge_lora.yaml
```

合并后的模型将保存在配置文件指定的输出目录。

### 4. 推理

#### 方式一：批量推理（vLLM）

使用 `inference.py` 进行高性能批量推理：

```bash
# 单张图片推理
python inference.py \
  --model_path /path/to/merged/model \
  --image /path/to/image.jpg

# 批量推理
python inference.py \
  --model_path /path/to/merged/model \
  --image_dir /path/to/images/ \
  --batch_size 32 \
  --output predictions.json
```

**参数说明**：
- `--model_path`: 合并后的模型路径
- `--image`: 单张图片路径
- `--image_dir`: 图片目录（批量推理）
- `--batch_size`: 批处理大小（默认 32，单 A10 建议 32-64）
- `--gpu_memory_utilization`: GPU 内存利用率（默认 0.9）
- `--output`: 输出 JSON 文件路径

#### 方式二：FastAPI 服务（OSS）

启动 FastAPI 推理服务，支持从阿里云 OSS 读取图片：

```bash
# 确保已配置 .env 文件中的 OSS 凭证
python api.py
```

服务启动后：
- **API 文档**: http://localhost:8888/docs
- **健康检查**: http://localhost:8888/health

**API 示例**：

```bash
# 单张图片推理
curl -X POST "http://localhost:8888/predict/oss" \
  -H "Content-Type: application/json" \
  -d '{
    "object_key": "products/image001.jpg",
    "image_name": "产品001"
  }'

# 批量推理
curl -X POST "http://localhost:8888/predict/batch/oss" \
  -H "Content-Type: application/json" \
  -d '{
    "object_keys": [
      "products/image001.jpg",
      "products/image002.jpg"
    ]
  }'
```

## 性能优化

- **vLLM**: 使用 PagedAttention 实现高吞吐量批量推理
- **LoRA**: 相比全量微调节省约 90% 显存
- **内网 OSS**: 使用阿里云内网 endpoint 节省流量费用并提高速度
- **批量处理**: 支持批量推理，单 A10 可达 32-64 batch size

## 注意事项

1. **数据集文件**: `data/` 目录已被 `.gitignore` 忽略，不会上传到 Git 仓库
2. **敏感信息**: OSS 凭证等敏感信息应存储在 `.env` 文件中，不要提交到代码库
3. **模型文件**: 合并后的模型文件较大，建议单独存储和管理
4. **GPU 显存**: Qwen2.5-VL-3B + LoRA 推理至少需要 16GB 显存

## 依赖项

主要依赖：
- `transformers>=4.47.0`
- `vllm>=0.6.5`
- `fastapi>=0.100.0`
- `oss2>=2.18.0`
- `torch>=2.1.0`
- `Pillow>=10.0.0`

完整依赖列表请参考 `requirements.txt`

## License

MIT License

## 致谢

- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) - 优秀的 LLM 微调框架
- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2-VL) - 强大的多模态大模型

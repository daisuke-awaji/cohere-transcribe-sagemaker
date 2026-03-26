# Cohere Transcribe on SageMaker

[Cohere Transcribe](https://cohere.com/blog/transcribe) を AWS SageMaker Real-time Endpoint にデプロイする CDK プロジェクト。

## モデル概要

| 項目 | 値 |
|---|---|
| モデル | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) |
| パラメータ数 | 2B |
| アーキテクチャ | Conformer (encoder-decoder) |
| ライセンス | Apache 2.0 |
| 対応言語 | 14言語 (en, ja, zh, ko, fr, de, it, es, pt, el, nl, pl, vi, ar) |

## 前提条件

- Node.js 20+
- AWS CDK CLI (`npm install -g aws-cdk`)
- Docker (CDK bundling に必要)
- HuggingFace アカウント + Cohere Transcribe への[アクセス申請](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)

## デプロイ

```bash
cd packages/cdk
npm install

# デプロイ (HuggingFace token を指定)
npx cdk deploy -c huggingFaceToken=hf_xxxxx

# オプション: リージョン・インスタンスタイプの変更
npx cdk deploy \
  -c huggingFaceToken=hf_xxxxx \
  -c instanceType=ml.g5.2xlarge \
  -c endpointName=my-transcribe-endpoint
```

> **注意**: モデルのダウンロード（約4GB）と初期化に 10〜15 分かかります。

## 使い方

### Python (boto3)

```python
import boto3
import json
import base64

client = boto3.client("sagemaker-runtime")

with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = client.invoke_endpoint(
    EndpointName="cohere-transcribe-endpoint",
    ContentType="application/json",
    Accept="application/json",
    Body=json.dumps({
        "audio_base64": audio_b64,
        "language": "ja"
    }),
)

result = json.loads(response["Body"].read())
print(result["transcription"])
```

### AWS CLI

```bash
# 音声ファイルを base64 エンコードして送信
AUDIO_B64=$(base64 -w0 audio.wav)

aws sagemaker-runtime invoke-endpoint \
  --endpoint-name cohere-transcribe-endpoint \
  --content-type "application/json" \
  --accept "application/json" \
  --body "{\"audio_base64\":\"${AUDIO_B64}\",\"language\":\"ja\"}" \
  output.json

cat output.json
```

## 対応入力形式

| Content-Type | 形式 |
|---|---|
| `application/json` | `{"audio_base64": "<base64>", "language": "ja"}` |
| `audio/wav` | WAV バイナリ (言語は環境変数 `TRANSCRIBE_LANGUAGE` のデフォルト値を使用) |

## コスト

- `ml.g5.xlarge` (NVIDIA A10G 24GB): 約 $1.408/hr (ap-northeast-1)
- 不要時は `npx cdk destroy` で削除してください

## クリーンアップ

```bash
cd packages/cdk
npx cdk destroy
```

## 構成

```
packages/
├── cdk/              # CDK インフラコード
│   ├── bin/app.ts
│   ├── lib/cohere-transcribe-stack.ts
│   └── cdk.json
└── inference/        # SageMaker 推論コード
    └── code/
        ├── inference.py
        └── requirements.txt
```

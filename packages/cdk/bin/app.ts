#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { CohereTranscribeStack } from "../lib/cohere-transcribe-stack";
import { VllmStreamingStack } from "../lib/vllm-streaming-stack";

const app = new cdk.App();

const huggingFaceToken = app.node.tryGetContext("huggingFaceToken");

if (!huggingFaceToken) {
  throw new Error(
    "huggingFaceToken is required. Pass it via: cdk deploy -c huggingFaceToken=hf_xxx"
  );
}

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION,
};

// --- SageMaker Endpoint (batch/request-response) ---
const modelId = app.node.tryGetContext("modelId") ?? "CohereLabs/cohere-transcribe-03-2026";
const instanceType = app.node.tryGetContext("instanceType") ?? "ml.g5.xlarge";
const endpointName = app.node.tryGetContext("endpointName") ?? "cohere-transcribe-endpoint";

new CohereTranscribeStack(app, "CohereTranscribeStack", {
  modelId,
  instanceType,
  endpointName,
  huggingFaceToken,
  env,
});

// --- vLLM + WebSocket Streaming (real-time) ---
new VllmStreamingStack(app, "VllmStreamingStack", {
  huggingFaceToken,
  env,
});


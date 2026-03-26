#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { CohereTranscribeStack } from "../lib/cohere-transcribe-stack";

const app = new cdk.App();

const modelId = app.node.tryGetContext("modelId") ?? "CohereLabs/cohere-transcribe-03-2026";
const instanceType = app.node.tryGetContext("instanceType") ?? "ml.g5.xlarge";
const endpointName = app.node.tryGetContext("endpointName") ?? "cohere-transcribe-endpoint";
const huggingFaceToken = app.node.tryGetContext("huggingFaceToken");

if (!huggingFaceToken) {
  throw new Error(
    "huggingFaceToken is required. Pass it via: cdk deploy -c huggingFaceToken=hf_xxx"
  );
}

new CohereTranscribeStack(app, "CohereTranscribeStack", {
  modelId,
  instanceType,
  endpointName,
  huggingFaceToken,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});

import * as path from "path";
import * as child_process from "child_process";
import {
  Stack,
  StackProps,
  CfnOutput,
  Aws,
  BundlingOutput,
  DockerImage,
  ILocalBundling,
} from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3assets from "aws-cdk-lib/aws-s3-assets";
import * as sagemaker from "aws-cdk-lib/aws-sagemaker";
import { Construct } from "constructs";

const DLC_ACCOUNT_ID = "763104351884";
const DLC_IMAGE_TAG =
  "2.6.0-transformers4.51.3-gpu-py312-cu124-ubuntu22.04";

interface CohereTranscribeStackProps extends StackProps {
  readonly modelId: string;
  readonly instanceType: string;
  readonly endpointName: string;
  readonly huggingFaceToken: string;
}

export class CohereTranscribeStack extends Stack {
  constructor(
    scope: Construct,
    id: string,
    props: CohereTranscribeStackProps
  ) {
    super(scope, id, props);

    const { modelId, instanceType, endpointName, huggingFaceToken } = props;

    const inferenceDir = path.join(__dirname, "../../inference");

    // --- Model Artifact (model.tar.gz) ---
    const modelArtifact = new s3assets.Asset(this, "ModelArtifact", {
      path: inferenceDir,
      bundling: {
        local: {
          tryBundle(outputDir: string): boolean {
            child_process.execSync(
              `tar czf ${outputDir}/model.tar.gz -C ${inferenceDir} code/`,
              { stdio: "inherit" }
            );
            return true;
          },
        } satisfies ILocalBundling,
        image: DockerImage.fromRegistry("public.ecr.aws/docker/library/alpine:3.20"),
        command: [
          "sh",
          "-c",
          "cd /asset-input && tar czf /asset-output/model.tar.gz code/",
        ],
        outputType: BundlingOutput.NOT_ARCHIVED,
      },
    });

    // --- IAM Role for SageMaker ---
    const executionRole = new iam.Role(this, "SageMakerExecutionRole", {
      assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "AmazonSageMakerFullAccess"
        ),
      ],
    });

    modelArtifact.grantRead(executionRole);

    // --- DLC Image URI ---
    const imageUri = [
      `${DLC_ACCOUNT_ID}.dkr.ecr.${Aws.REGION}.amazonaws.com`,
      `huggingface-pytorch-inference:${DLC_IMAGE_TAG}`,
    ].join("/");

    // --- SageMaker Model ---
    const model = new sagemaker.CfnModel(this, "TranscribeModel", {
      executionRoleArn: executionRole.roleArn,
      primaryContainer: {
        image: imageUri,
        modelDataUrl: modelArtifact.s3ObjectUrl,
        environment: {
          HF_MODEL_ID: modelId,
          HF_TOKEN: huggingFaceToken,
          HF_TRUST_REMOTE_CODE: "1",
          SAGEMAKER_PROGRAM: "inference.py",
          SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
          TRANSCRIBE_LANGUAGE: "ja",
        },
      },
    });

    // --- Endpoint Configuration ---
    const endpointConfig = new sagemaker.CfnEndpointConfig(
      this,
      "TranscribeEndpointConfig",
      {
        productionVariants: [
          {
            initialInstanceCount: 1,
            instanceType,
            modelName: model.attrModelName,
            variantName: "Primary",
            initialVariantWeight: 1,
            containerStartupHealthCheckTimeoutInSeconds: 900,
            modelDataDownloadTimeoutInSeconds: 1800,
          },
        ],
      }
    );

    // --- Endpoint ---
    const endpoint = new sagemaker.CfnEndpoint(
      this,
      "TranscribeEndpoint",
      {
        endpointConfigName: endpointConfig.attrEndpointConfigName,
        endpointName,
      }
    );

    // --- Outputs ---
    new CfnOutput(this, "EndpointName", {
      value: endpoint.endpointName!,
      description: "SageMaker Endpoint Name",
    });

    new CfnOutput(this, "ModelName", {
      value: model.attrModelName,
      description: "SageMaker Model Name",
    });

    new CfnOutput(this, "InvokeCommand", {
      value: [
        "aws sagemaker-runtime invoke-endpoint",
        `--endpoint-name ${endpointName}`,
        '--content-type "application/json"',
        '--accept "application/json"',
        '--body \'{"audio_base64":"<BASE64_AUDIO>","language":"ja"}\'',
        "output.json",
      ].join(" "),
      description: "Example invoke command",
    });
  }
}

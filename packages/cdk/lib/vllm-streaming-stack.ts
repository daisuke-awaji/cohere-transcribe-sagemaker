import * as fs from "fs";
import * as path from "path";
import {
  Stack,
  StackProps,
  CfnOutput,
  Duration,
  Tags,
} from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

const DLAMI_NAME_PATTERN =
  "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*";

interface VllmStreamingStackProps extends StackProps {
  readonly huggingFaceToken: string;
  readonly instanceType?: string;
}

export class VllmStreamingStack extends Stack {
  constructor(
    scope: Construct,
    id: string,
    props: VllmStreamingStackProps
  ) {
    super(scope, id, props);

    const instanceType =
      props.instanceType ?? "g6.xlarge";

    // --- VPC ---
    const vpc = new ec2.Vpc(this, "Vpc", {
      natGateways: 0,
      subnetConfiguration: [
        {
          name: "Public",
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
      ],
      availabilityZones: [`${this.region}c`],
    });

    // --- Security Group ---
    const sg = new ec2.SecurityGroup(this, "InstanceSG", {
      vpc,
      description: "vLLM + WebSocket server",
      allowAllOutbound: true,
    });
    sg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(22), "SSH");
    sg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(8000), "vLLM REST API");
    sg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(8765), "WebSocket");

    // --- IAM Role ---
    const role = new iam.Role(this, "InstanceRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "AmazonSSMManagedInstanceCore"
        ),
      ],
    });

    // --- Key Pair (use existing or create) ---
    const keyPair = new ec2.KeyPair(this, "KeyPair", {
      keyPairName: "cohere-transcribe-keypair",
    });

    // --- Deep Learning AMI ---
    const ami = ec2.MachineImage.lookup({
      name: DLAMI_NAME_PATTERN,
      owners: ["amazon"],
    });

    // --- Build UserData ---
    const streamingDir = path.join(__dirname, "../../streaming");
    const wsServerContent = fs.readFileSync(
      path.join(streamingDir, "websocket_server.py"),
      "utf-8"
    );
    let setupScript = fs.readFileSync(
      path.join(streamingDir, "setup.sh"),
      "utf-8"
    );
    setupScript = setupScript.replace("__HF_TOKEN__", props.huggingFaceToken);
    setupScript = setupScript.replace(
      "__WEBSOCKET_SERVER_CONTENT__",
      wsServerContent
    );

    const userData = ec2.UserData.forLinux();
    userData.addCommands(setupScript);

    // --- EC2 Instance ---
    const instance = new ec2.Instance(this, "VllmInstance", {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      instanceType: new ec2.InstanceType(instanceType),
      machineImage: ami,
      securityGroup: sg,
      role,
      keyPair,
      userData,
      blockDevices: [
        {
          deviceName: "/dev/sda1",
          volume: ec2.BlockDeviceVolume.ebs(100, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            encrypted: true,
          }),
        },
      ],
      associatePublicIpAddress: true,
    });

    Tags.of(instance).add("Name", "CohereTranscribe-vLLM");

    // --- Outputs ---
    new CfnOutput(this, "InstanceId", {
      value: instance.instanceId,
      description: "EC2 Instance ID",
    });

    new CfnOutput(this, "PublicIp", {
      value: instance.instancePublicIp,
      description: "EC2 Public IP",
    });

    new CfnOutput(this, "VllmEndpoint", {
      value: `http://${instance.instancePublicIp}:8000`,
      description: "vLLM REST API endpoint",
    });

    new CfnOutput(this, "WebSocketEndpoint", {
      value: `ws://${instance.instancePublicIp}:8765`,
      description: "WebSocket streaming endpoint",
    });

    new CfnOutput(this, "SSHCommand", {
      value: `ssh -i ~/.ssh/${keyPair.keyPairName}.pem ubuntu@${instance.instancePublicIp}`,
      description: "SSH command",
    });

    new CfnOutput(this, "SetupLogCommand", {
      value: `aws ssm start-session --target ${instance.instanceId} --document-name AWS-StartInteractiveCommand --parameters command="journalctl -u vllm -f"`,
      description: "View vLLM logs via SSM",
    });
  }
}

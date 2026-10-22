from pathlib import Path

import aws_cdk as cdk
from aws_cdk import (
    Stack, Duration, RemovalPolicy, CfnOutput,
    aws_dynamodb as dynamodb,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
)
from constructs import Construct

# This file lives in the project root, so the root is just its own folder
PROJECT_ROOT = str(Path(__file__).resolve().parent)


class NidsStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # DynamoDB table that stores the predictions
        table = dynamodb.Table(
            self, "PredictionsTable",
            partition_key=dynamodb.Attribute(
                name="prediction_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,  # fine for a project; remove for production
        )

        # Lambda function built from the Dockerfile in the project root
        fn = _lambda.DockerImageFunction(
            self, "NidsFunction",
            code=_lambda.DockerImageCode.from_image_asset(
                PROJECT_ROOT,
                exclude=["data", "reports", ".venv", "venv", ".git", "cdk.out"],
            ),
            architecture=_lambda.Architecture.ARM_64,  # Apple Silicon Mac; use X86_64 on an Intel Mac
            memory_size=2048,
            timeout=Duration.seconds(30),
            environment={"DYNAMODB_TABLE_NAME": table.table_name},
        )

        # Allow the Lambda to write to the table
        table.grant_write_data(fn)

        # API Gateway that sends every request to FastAPI
        api = apigw.LambdaRestApi(self, "NidsApi", handler=fn)

        CfnOutput(self, "ApiUrl", value=api.url)


# Entry point: this is what the cdk command runs
app = cdk.App()
NidsStack(app, "NidsStack")
app.synth()

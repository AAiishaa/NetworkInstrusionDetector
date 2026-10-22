from pathlib import Path
from aws_cdk import (
    Stack, Duration, RemovalPolicy, CfnOutput,
    aws_dynamodb as dynamodb,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
)
from constructs import Construct

# infra/infra/infra_stack.py -> parents[2] is the project root
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


class InfraStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # DynamoDB table that stores the predictions
        table = dynamodb.Table(
            self, "PredictionsTable",
            partition_key=dynamodb.Attribute(
                name="prediction_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,  # cdk destroy deletes the table too
        )

        # Lambda function built from your Dockerfile
        fn = _lambda.DockerImageFunction(
            self, "NidsFunction",
            code=_lambda.DockerImageCode.from_image_asset(
                PROJECT_ROOT,
                exclude=["data", "infra", "reports", ".venv", ".git", "cdk.out"],
            ),
            architecture=_lambda.Architecture.ARM_64,  # matches your arm64 Mac
            memory_size=2048,
            timeout=Duration.seconds(30),
            environment={"DYNAMODB_TABLE_NAME": table.table_name},
        )

        # Lets the Lambda write to the table
        table.grant_write_data(fn)

        # API Gateway that sends every request to FastAPI
        api = apigw.LambdaRestApi(self, "NidsApi", handler=fn)

        # Prints your API address after deploying
        CfnOutput(self, "ApiUrl", value=api.url)
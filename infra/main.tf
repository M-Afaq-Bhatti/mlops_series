terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name, used as a prefix for resource names"
  type        = string
  default     = "student-score-predictor"
}

variable "monthly_budget_usd" {
  description = "Hard monthly spend limit - set well under the $40 credit"
  type        = string
  default     = "20"
}

variable "alert_email" {
  description = "Email to notify when budget thresholds are hit"
  type        = string
}

# --- S3: model artifacts and DVC remote storage ---
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.project_name}-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_lifecycle_configuration" "expire_old_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  rule {
    id     = "expire-old-model-versions"
    status = "Enabled"

    filter {
      prefix = "models/"
    }

    expiration {
      days = 90
    }
  }
}

data "aws_caller_identity" "current" {}

# --- ECR: Docker image registry for the serving container ---
resource "aws_ecr_repository" "score_model_repo" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "expire_untagged" {
  repository = aws_ecr_repository.score_model_repo.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire untagged images after 14 days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 14
      }
      action = { type = "expire" }
    }]
  })
}

# --- IAM: least-privilege role for the Lambda serving function ---
resource "aws_iam_role" "lambda_exec_role" {
  name = "${var.project_name}-lambda-exec-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_s3_read" {
  name = "${var.project_name}-lambda-s3-read"
  role = aws_iam_role.lambda_exec_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject"]
      Resource = "${aws_s3_bucket.model_artifacts.arn}/*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_logging" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# --- Budget guard: set BEFORE anything else touches the account ---
resource "aws_budgets_budget" "cost_guard" {
  name         = "${var.project_name}-monthly-limit"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_usd
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}

output "s3_bucket_name" {
  value = aws_s3_bucket.model_artifacts.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.score_model_repo.repository_url
}

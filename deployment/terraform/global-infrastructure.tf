# Global multi-region infrastructure for Materials Orchestrator
# Supports AWS, GCP, and Azure deployment

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
  
  backend "s3" {
    bucket = "materials-orchestrator-terraform-state"
    key    = "global/terraform.tfstate"
    region = "us-west-2"
  }
}

# Variables
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "materials-orchestrator"
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "regions" {
  description = "List of regions to deploy to"
  type = object({
    aws = list(string)
    gcp = list(string)
    azure = list(string)
  })
  default = {
    aws   = ["us-west-2", "us-east-1", "eu-west-1", "ap-southeast-1"]
    gcp   = ["us-central1", "europe-west1", "asia-southeast1"]
    azure = ["West US 2", "East US", "West Europe", "Southeast Asia"]
  }
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

# Local values
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    CreatedAt   = formatdate("YYYY-MM-DD", timestamp())
  }
}

# AWS Multi-Region Deployment
module "aws_infrastructure" {
  source = "./modules/aws"
  
  for_each = toset(var.regions.aws)
  
  project_name       = var.project_name
  environment        = var.environment
  region            = each.key
  kubernetes_version = var.kubernetes_version
  
  # EKS Configuration
  cluster_config = {
    node_groups = {
      general = {
        instance_types = ["m5.large", "m5.xlarge"]
        capacity_type  = "ON_DEMAND"
        scaling_config = {
          desired_size = 3
          max_size     = 20
          min_size     = 1
        }
      }
      spot = {
        instance_types = ["m5.large", "m5.xlarge", "c5.large", "c5.xlarge"]
        capacity_type  = "SPOT"
        scaling_config = {
          desired_size = 2
          max_size     = 10
          min_size     = 0
        }
      }
    }
  }
  
  # Database Configuration
  database_config = {
    engine_version    = "7.0"
    instance_class    = "db.r6g.large"
    allocated_storage = 100
    multi_az          = true
    backup_retention  = 30
  }
  
  # Redis Configuration
  redis_config = {
    node_type           = "cache.r6g.large"
    num_cache_nodes     = 3
    engine_version      = "7.0"
    port                = 6379
    parameter_group     = "default.redis7"
  }
  
  tags = local.common_tags
}

# GCP Multi-Region Deployment  
module "gcp_infrastructure" {
  source = "./modules/gcp"
  
  for_each = toset(var.regions.gcp)
  
  project_name       = var.project_name
  environment        = var.environment
  region            = each.key
  kubernetes_version = var.kubernetes_version
  
  # GKE Configuration
  cluster_config = {
    node_pools = {
      general = {
        machine_type   = "e2-standard-4"
        disk_size_gb   = 100
        disk_type      = "pd-ssd"
        preemptible    = false
        min_node_count = 1
        max_node_count = 20
        initial_count  = 3
      }
      spot = {
        machine_type   = "e2-standard-4"
        disk_size_gb   = 100
        disk_type      = "pd-standard"
        preemptible    = true
        min_node_count = 0
        max_node_count = 10
        initial_count  = 2
      }
    }
  }
  
  # Database Configuration (Cloud SQL)
  database_config = {
    tier                = "db-custom-4-16384"
    disk_size           = 100
    disk_type           = "PD_SSD"
    backup_enabled      = true
    binary_log_enabled  = true
    availability_type   = "REGIONAL"
  }
  
  # Redis Configuration (Memorystore)
  redis_config = {
    memory_size_gb = 4
    tier           = "STANDARD_HA"
    version        = "REDIS_7_0"
  }
  
  labels = local.common_tags
}

# Azure Multi-Region Deployment
module "azure_infrastructure" {
  source = "./modules/azure"
  
  for_each = toset(var.regions.azure)
  
  project_name       = var.project_name
  environment        = var.environment
  location          = each.key
  kubernetes_version = var.kubernetes_version
  
  # AKS Configuration
  cluster_config = {
    node_pools = {
      general = {
        vm_size         = "Standard_D4s_v3"
        os_disk_size_gb = 100
        type            = "VirtualMachineScaleSets"
        min_count       = 1
        max_count       = 20
        node_count      = 3
      }
      spot = {
        vm_size         = "Standard_D4s_v3"
        os_disk_size_gb = 100
        priority        = "Spot"
        eviction_policy = "Delete"
        min_count       = 0
        max_count       = 10
        node_count      = 2
      }
    }
  }
  
  # Database Configuration (Azure Database for PostgreSQL)
  database_config = {
    sku_name                     = "GP_Gen5_4"
    storage_mb                   = 102400
    backup_retention_days        = 30
    geo_redundant_backup_enabled = true
    auto_grow_enabled            = true
  }
  
  # Redis Configuration (Azure Cache for Redis)
  redis_config = {
    capacity = 2
    family   = "C"
    sku_name = "Standard"
    version  = "6"
  }
  
  tags = local.common_tags
}

# Global Load Balancer and CDN
resource "aws_cloudfront_distribution" "global_cdn" {
  comment             = "${var.project_name} Global CDN"
  default_root_object = "index.html"
  enabled             = true
  is_ipv6_enabled     = true
  price_class         = "PriceClass_All"
  
  # Origins for each region
  dynamic "origin" {
    for_each = var.regions.aws
    content {
      domain_name = module.aws_infrastructure[origin.value].load_balancer_dns
      origin_id   = "aws-${origin.value}"
      
      custom_origin_config {
        http_port              = 80
        https_port             = 443
        origin_protocol_policy = "https-only"
        origin_ssl_protocols   = ["TLSv1.2"]
      }
    }
  }
  
  # Default cache behavior
  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "aws-${var.regions.aws[0]}"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
    
    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Content-Type"]
      
      cookies {
        forward = "none"
      }
    }
    
    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }
  
  # Geographic restrictions
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  # SSL Certificate
  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.global_cert.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
  
  tags = local.common_tags
}

# Global SSL Certificate
resource "aws_acm_certificate" "global_cert" {
  provider          = aws.us_east_1
  domain_name       = "materials.terragonlabs.com"
  validation_method = "DNS"
  
  subject_alternative_names = [
    "*.materials.terragonlabs.com",
    "dashboard.materials.terragonlabs.com",
    "api.materials.terragonlabs.com"
  ]
  
  lifecycle {
    create_before_destroy = true
  }
  
  tags = local.common_tags
}

# Global DNS with Route53
resource "aws_route53_zone" "main" {
  name = "materials.terragonlabs.com"
  
  tags = local.common_tags
}

resource "aws_route53_record" "main" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "materials.terragonlabs.com"
  type    = "A"
  
  alias {
    name                   = aws_cloudfront_distribution.global_cdn.domain_name
    zone_id                = aws_cloudfront_distribution.global_cdn.hosted_zone_id
    evaluate_target_health = false
  }
}

# Health check records for each region
resource "aws_route53_health_check" "regional_health" {
  for_each = toset(var.regions.aws)
  
  fqdn                            = module.aws_infrastructure[each.key].load_balancer_dns
  port                            = 443
  type                            = "HTTPS"
  resource_path                   = "/health"
  failure_threshold               = "3"
  request_interval                = "30"
  cloudwatch_logs_region          = each.key
  cloudwatch_logs_log_group_name  = "/aws/route53/${var.project_name}-${each.key}"
  
  tags = merge(local.common_tags, {
    Region = each.key
  })
}

# Global monitoring and alerting
resource "aws_sns_topic" "global_alerts" {
  name = "${var.project_name}-global-alerts"
  
  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "global_health" {
  for_each = toset(var.regions.aws)
  
  alarm_name          = "${var.project_name}-${each.key}-health"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "HealthCheckStatus"
  namespace           = "AWS/Route53"
  period              = "60"
  statistic           = "Minimum"
  threshold           = "1"
  alarm_description   = "Regional health check for ${each.key}"
  alarm_actions       = [aws_sns_topic.global_alerts.arn]
  
  dimensions = {
    HealthCheckId = aws_route53_health_check.regional_health[each.key].id
  }
  
  tags = local.common_tags
}

# Outputs
output "global_endpoints" {
  description = "Global endpoints"
  value = {
    main_domain = aws_route53_record.main.fqdn
    cdn_domain  = aws_cloudfront_distribution.global_cdn.domain_name
  }
}

output "regional_endpoints" {
  description = "Regional endpoints"
  value = {
    aws   = { for k, v in module.aws_infrastructure : k => v.cluster_endpoint }
    gcp   = { for k, v in module.gcp_infrastructure : k => v.cluster_endpoint }
    azure = { for k, v in module.azure_infrastructure : k => v.cluster_endpoint }
  }
}

output "monitoring" {
  description = "Monitoring endpoints"
  value = {
    cloudwatch_dashboard = "https://console.aws.amazon.com/cloudwatch/home"
    sns_topic_arn       = aws_sns_topic.global_alerts.arn
    health_checks       = { for k, v in aws_route53_health_check.regional_health : k => v.id }
  }
}
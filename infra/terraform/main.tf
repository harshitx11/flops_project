terraform {
  required_version = ">= 1.5"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.35"
    }
  }
}

provider "kubernetes" {
  config_path    = var.kubeconfig_path
  config_context = var.kube_context
}

resource "kubernetes_namespace" "sentiment" {
  metadata {
    name = var.namespace
    labels = {
      app = "sentiment-api"
    }
  }
}

resource "kubernetes_resource_quota" "sentiment" {
  metadata {
    name      = "sentiment-quota"
    namespace = kubernetes_namespace.sentiment.metadata[0].name
  }
  spec {
    hard = {
      pods            = "6"
      "limits.cpu"    = "2"
      "limits.memory" = "2Gi"
    }
  }
}

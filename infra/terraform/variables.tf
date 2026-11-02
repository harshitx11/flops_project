variable "kubeconfig_path" {
  description = "Path to the kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "Kubernetes context to deploy to"
  type        = string
  default     = null
}

variable "namespace" {
  description = "Namespace for the sentiment API"
  type        = string
  default     = "sentiment"
}

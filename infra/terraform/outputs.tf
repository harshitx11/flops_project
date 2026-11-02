output "namespace" {
  description = "Namespace created for the sentiment API"
  value       = kubernetes_namespace.sentiment.metadata[0].name
}

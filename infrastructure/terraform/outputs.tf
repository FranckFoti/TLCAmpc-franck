output "db_public_ip" {
  description = "Public IP of the database server"
  value       = hcloud_server.db.ipv4_address
}

output "db_private_ip" {
  description = "Private IP of the database server (internal network)"
  value       = one(hcloud_server.db.network[*].ip)
}

output "db_connection_string" {
  description = "PostgreSQL connection string (use from worker network or via SSH tunnel)"
  value       = "postgresql://dronesim:PASSWORD@${one(hcloud_server.db.network[*].ip)}:5432/dronesim"
  sensitive   = true
}

output "worker_public_ips" {
  description = "Public IPs of all worker servers"
  value       = hcloud_server.worker[*].ipv4_address
}

output "worker_count" {
  description = "Number of worker servers provisioned"
  value       = var.worker_count
}

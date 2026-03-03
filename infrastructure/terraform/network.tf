data "hcloud_ssh_key" "default" {
  name = "linda.klesper@gmx.de"
}

resource "hcloud_network" "internal" {
  name     = "dronesim-net"
  ip_range = "10.0.0.0/16"
}

resource "hcloud_network_subnet" "subnet" {
  network_id   = hcloud_network.internal.id
  type         = "cloud"
  network_zone = "eu-central"
  ip_range     = "10.0.1.0/24"
}

# Firewall for the DB server: PostgreSQL only from private network, SSH from anywhere
resource "hcloud_firewall" "db" {
  name = "dronesim-db-fw"

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "5432"
    source_ips = ["10.0.0.0/16"]
  }
}

# Firewall for worker servers: SSH only
resource "hcloud_firewall" "worker" {
  name = "dronesim-worker-fw"

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = ["0.0.0.0/0", "::/0"]
  }
}

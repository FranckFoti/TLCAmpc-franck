resource "hcloud_server" "db" {
  name        = "dronesim-db"
  server_type = var.db_server_type
  location    = var.location
  image       = "ubuntu-24.04"
  ssh_keys    = [data.hcloud_ssh_key.default.id]
  firewall_ids = [hcloud_firewall.db.id]

  user_data = templatefile("${path.module}/cloud-init-db.yaml.tpl", {
    db_password = var.db_password
  })

  network {
    network_id = hcloud_network.internal.id
  }

  depends_on = [hcloud_network_subnet.subnet]
}

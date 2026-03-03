resource "hcloud_server" "worker" {
  count       = var.worker_count
  name        = "dronesim-worker-${count.index}"
  server_type = var.worker_server_type
  location    = var.location
  image       = "ubuntu-24.04"
  ssh_keys    = [data.hcloud_ssh_key.default.id]
  firewall_ids = [hcloud_firewall.worker.id]

  user_data = templatefile("${path.module}/cloud-init-worker.yaml.tpl", {
    db_password        = var.db_password
    db_private_ip      = one(hcloud_server.db.network[*].ip)
    deploy_key_private = var.deploy_key_private
    git_repo_url       = var.git_repo_url
    git_branch         = var.git_branch
    v_max              = var.scenario_v_max
    u_max              = var.scenario_u_max
    room_size          = var.scenario_room_size
    static_safety_zone = var.scenario_static_safety_zone
    adaptive_safety_zone = var.scenario_adaptive_safety_zone
    r_min              = var.scenario_r_min
  })

  network {
    network_id = hcloud_network.internal.id
  }

  depends_on = [hcloud_network_subnet.subnet, hcloud_server.db]
}

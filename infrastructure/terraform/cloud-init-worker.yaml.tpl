#cloud-config

package_update: true
package_upgrade: true

packages:
  - python3
  - python3-pip
  - python3-venv
  - git
  - build-essential
  - libpq-dev

write_files:
  - path: /root/.ssh/deploy_key
    permissions: "0600"
    content: |
      ${deploy_key_private}

  - path: /root/.ssh/config
    permissions: "0600"
    content: |
      Host github.com
        IdentityFile /root/.ssh/deploy_key
        StrictHostKeyChecking no

  - path: /etc/systemd/system/sim-worker.service
    content: |
      [Unit]
      Description=Drone Simulation Worker
      After=network-online.target
      Wants=network-online.target

      [Service]
      Type=simple
      User=root
      WorkingDirectory=/opt/TLCAmpc
      Environment=DRONESIM_DATABASE_URL=postgresql://dronesim:${db_password}@${db_private_ip}:5432/dronesim
      ExecStart=/opt/TLCAmpc/.venv/bin/python -m paper2_tools.distributed.worker \
        --n-workers -1 \
        --v-max ${v_max} \
        --u-max ${u_max} \
        --room-size ${room_size} \
        --static-safety-zone ${static_safety_zone} \
        --adaptive-safety-zone ${adaptive_safety_zone} \
        --r-min ${r_min} \
        --log-level INFO
      Restart=on-failure
      RestartSec=30

      [Install]
      WantedBy=multi-user.target

runcmd:
  - GIT_SSH_COMMAND="ssh -i /root/.ssh/deploy_key -o StrictHostKeyChecking=no" git clone -b ${git_branch} ${git_repo_url} /opt/TLCAmpc
  - python3 -m venv /opt/TLCAmpc/.venv
  - /opt/TLCAmpc/.venv/bin/pip install --upgrade pip
  - /opt/TLCAmpc/.venv/bin/pip install -e "/opt/TLCAmpc[distributed]"
  - /opt/TLCAmpc/.venv/bin/pip uninstall -y PySide6 PySide6-Essentials PySide6-Addons shiboken6
  - systemctl daemon-reload
  - systemctl enable sim-worker
  - systemctl start sim-worker

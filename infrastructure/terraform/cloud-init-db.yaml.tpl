#cloud-config

package_update: true
package_upgrade: true

packages:
  - postgresql-16
  - postgresql-client-16

write_files:
  - path: /etc/postgresql/16/main/conf.d/listen.conf
    content: |
      listen_addresses = '*'
      max_connections = 200
    owner: postgres:postgres

  - path: /tmp/setup-db.sql
    content: |
      CREATE USER dronesim WITH PASSWORD '${db_password}';
      CREATE DATABASE dronesim OWNER dronesim;
      GRANT ALL PRIVILEGES ON DATABASE dronesim TO dronesim;

  - path: /tmp/setup-pg-hba.sh
    permissions: "0755"
    content: |
      #!/bin/bash
      # Allow connections from private network
      echo "host dronesim dronesim 10.0.0.0/16 md5" >> /etc/postgresql/16/main/pg_hba.conf
      systemctl restart postgresql

runcmd:
  - systemctl enable postgresql
  - systemctl start postgresql
  - sleep 2
  - sudo -u postgres psql -f /tmp/setup-db.sql
  - bash /tmp/setup-pg-hba.sh
  - rm /tmp/setup-db.sql /tmp/setup-pg-hba.sh

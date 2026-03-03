#cloud-config

package_update: true
package_upgrade: true

packages:
  - postgresql
  - postgresql-client

runcmd:
  - |
    PG_VERSION=$(pg_lsclusters -h | awk '{print $1}' | head -1)
    echo "listen_addresses = '*'" >> /etc/postgresql/$PG_VERSION/main/conf.d/listen.conf
    echo "max_connections = 200" >> /etc/postgresql/$PG_VERSION/main/conf.d/listen.conf
    systemctl restart postgresql
    sleep 2
    sudo -u postgres psql -c "CREATE USER dronesim WITH PASSWORD '${db_password}';"
    sudo -u postgres psql -c "CREATE DATABASE dronesim OWNER dronesim;"
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE dronesim TO dronesim;"
    echo "host dronesim dronesim 10.0.0.0/16 md5" >> /etc/postgresql/$PG_VERSION/main/pg_hba.conf
    systemctl restart postgresql

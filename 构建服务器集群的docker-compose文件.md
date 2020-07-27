# 构建服务器集群的docker-compose文件

```
version: '3.7'

services:
    master-0:
        image: qiangzi-centos-7.8:v0.2
        container_name: master-0
        hostname: master-0
        privileged: true
        networks:
            db_net:
                ipv4_address: 172.19.0.100
        restart: always

    slave-0:
        image: qiangzi-centos-7.8:v0.2
        container_name: slave-0
        hostname: slave-0
        privileged: true
        networks:
            db_net:
                ipv4_address: 172.19.0.101
        restart: always

    slave-1:
        image: qiangzi-centos-7.8:v0.2
        container_name: slave-1
        hostname: slave-1
        privileged: true
        networks:
            db_net:
                ipv4_address: 172.19.0.102
        restart: always

    slave-2:
        image: qiangzi-centos-7.8:v0.2
        container_name: slave-2
        hostname: slave-2
        privileged: true
        networks:
            db_net:
                ipv4_address: 172.19.0.103
        restart: always

    slave-3:
        image: qiangzi-centos-7.8:v0.2
        container_name: slave-3
        hostname: slave-3
        privileged: true
        networks:
            db_net:
                ipv4_address: 172.19.0.104
        restart: always

networks:
    db_net:
        external: true
        name: db_net
```
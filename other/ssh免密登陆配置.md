# ssh免密登陆

## 生成密钥
ssh-keygen -t rsa

## 将公钥写入权限文件
cd ~/.ssh

cat id_rsa.pub >> authorized_keys 

## 将其他机器的公钥 发送到master-0机器上
scp id_rsa.pub root@master-0:/root/.ssh/id_rsa.pub-slave0

## 在master-0机器上,将其他机器的公钥写入到权限文件
cat id_rsa.pub-slave0 >> authorized_keys

## 在master-0机器上, 将填充所有机器公钥的权限文件 分发到各个服务器
scp authorized_keys root@slave-0:/root/.ssh/authorized_keys


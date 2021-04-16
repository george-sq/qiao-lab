# Centos7.8服务器镜像构建步骤

## 获取基础镜像
docker pull centos:7.8.2003

## 启动容器
docker run -itd --privileged=true --name="centos-7.8" centos:7.8.2003 /usr/sbin/init 

## 修改时区
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

## 更新yum源
mv /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.backup
curl -o /etc/yum.repos.d/CentOS-Base.repo https://mirrors.aliyun.com/repo/Centos-7.repo
sed -i -e '/mirrors.cloud.aliyuncs.com/d' -e '/mirrors.aliyuncs.com/d' /etc/yum.repos.d/CentOS-Base.repo
yum update
yum makecache

## 基础软件安装
yum install -y which wget vim tree less net-tools make gcc curl curl-devel gettext-devel openssl-devel openssl openssh-server httpd ntp

## 增加epel源
wget -O /etc/yum.repos.d/epel.repo http://mirrors.aliyun.com/repo/epel-7.repo
yum update
yum install -y axel expat-devel libcurl-devel perl-ExtUtils-MakeMaker package

## 配置ssh
vim /etc/ssh/sshd_config
修改如下配置:
```
# Port 22
PermitRootLogin yes

RSAAuthentication yes
PubkeyAuthentication yes
PasswordAuthentication yes
```

systemctl start sshd.service
systemctl enable sshd.service

systemctl start httpd.service
systemctl enable httpd.service

systemctl start ntpd.service
systemctl enable ntpd.service

ssh-keygen -t rsa

## 安装git
yum remove git

cd ~/downloads
axel https://mirrors.edge.kernel.org/pub/software/scm/git/git-2.27.0.tar.gz
tar -xzvf git-*.tar.gz
cd git-*

make prefix=/usr/local/git all
make prefix=/usr/local/git install

vim /etc/profile.d/git.sh
```
GIT_HOME="/usr/local/git"
PATH=$GIT_HOME:$GIT_HOME/bin:$PATH
```

## 安装java/maven
cd ~/downloads
axel https://code.aliyun.com/kar/ojdk8-8u261/raw/7d4b7c1585a53866e6494d4ea457a0f99e59f3a1/jdk-8u261-linux-x64.tar.gz
axel https://mirror.bit.edu.cn/apache/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz

tar -xzvf jdk-8u261-linux-x64.tar.gz
mv jdk1.8.0_261 /usr/local/
tar -xzvf apache-maven-3.6.3-bin.tar.gz
mv apache-maven-3.6.3 /usr/local/

vim /etc/profile.d/java.sh
```
JAVA_HOME="/usr/local/jdk1.8.0_261"
MAVEN_HOME="/usr/local/apache-maven-3.6.3"
PATH=$JAVA_HOME:$JAVA_HOME/bin:$MAVEN_HOME:$MAVEN_HOME/bin:$PATH
```

vim /usr/local/apache-maven-3.6.3/conf/settings.xml
```
<mirror>
    <id>aliyunmaven</id>
    <mirrorOf>*</mirrorOf>
    <name>aliyun-repo</name>
    <url>https://maven.aliyun.com/repository/public</url>
</mirror>
```

## 安装miniconda / pip
cd ~/downloads
axel https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh
vim /etc/profile.d/miniconda.sh
```
MINICONDA_HOME="/usr/local/miniconda3"
PATH=$MINICONDA_HOME:$MINICONDA_HOME/bin:$PATH
```

conda config --set show_channel_urls yes
vim ~/.condarc
```
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```
conda update --all -y
conda clean --all -y

mkdir ~/.pip
vim ~/.pip/pip.conf
```
[global]
index-url=http://mirrors.aliyun.com/pypi/simple/
## index-url=http://pypi.doubanio.com/simple/
## index-url=https://mirrors.ustc.edu.cn/pypi/web/simple/
## index-url=https://pypi.tuna.tsinghua.edu.cn/simple/

trusted-host=mirrors.aliyun.com
## trusted-host=pypi.doubanio.com
## trusted-host=mirrors.ustc.edu.cn
## trusted-host=pypi.tuna.tsinghua.edu.cn
```

## 修改root用户密码
echo "root123456" | passwd --stdin root

## 退出容器
exit

## 构建docker镜像
docker commit -a="dp-qiangzi" -m="Centos7.8服务器镜像" 容器ID 用户名/repo_name:tag

docker commit -a="dp-qiangzi" -m="Centos7.8服务器镜像" 5b56b7b2357a qiangzi-centos-7.8:v0.1





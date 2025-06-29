FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y \
      sudo git python3 python3-venv python3-pip python-is-python3 \
      openjdk-17-jdk wget gnupg2 apt-transport-https \
      ca-certificates curl unzip software-properties-common && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key \
  | gpg --dearmor -o /usr/share/keyrings/jenkins-keyring.gpg && \
  echo "deb [signed-by=/usr/share/keyrings/jenkins-keyring.gpg] \
  https://pkg.jenkins.io/debian-stable binary/" \
  > /etc/apt/sources.list.d/jenkins.list && \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y jenkins && \
  rm -rf /var/lib/apt/lists/*

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
    unzip /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/aws /tmp/awscliv2.zip

VOLUME ["/var/lib/jenkins"] 
EXPOSE 8080 50000

RUN echo "jenkins ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /var/lib/jenkins/.wdm \
 && chown jenkins:jenkins /var/lib/jenkins/.wdm

USER jenkins
ENTRYPOINT ["java", "-Dfile.encoding=UTF-8", "-jar", "/usr/share/java/jenkins.war"]
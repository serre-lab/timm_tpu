sudo mkdir timmtpuworkspace/
sudo chmod 777 timmtpuworkspace/
cd timmtpuworkspace/
sudo mkdir /mnt/disks
sudo mkdir /mnt/disks/imagenet
sudo mount -o discard,defaults,noload  /dev/sdb /mnt/disks/imagenet/

pip install timm wandb

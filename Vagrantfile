Vagrant.configure("2") do |config|
    config.vm.box = "ubuntu/focal64"
    config.vm.box_version = "~> 20221202.0.1"
    config.vm.boot_timeout = 600

    config.vm.network "forwarded_port", guest: 4040, host: 4040

    config.vm.provision "shell", inline: <<-SHELL
        systemctl disable apt-daily.service
        systemctl disable apt-daily.timer

        sudo apt-get update -y
        sudo apt-get install python3-venv python3-opencv zip -y

        touch /home/vagrant/.bash_aliases
        if ! grep -q PTYHON_ALIAS_ADDED /home/vagrant/.bash_aliases; then
            echo "# PYTHON_ALIAS_ADDED" >> /home/vagrant/.bash_aliases
            echo "alias python='python3'" >> /home/vagrant/.bash_aliases
        fi
    SHELL
end
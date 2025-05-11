sudo apt install libglm-dev libglfw3-dev libfreetype-dev # libsoil-dev libassimp-dev libglew-dev libxinerama-dev libxcursor-dev  libxi-dev libgl1-mesa-dev xorg-dev

# display under wsl
export DISPLAY=$(grep nameserver /etc/resolv.conf | sed 's/nameserver //' ):0
export LIBGL_ALWAYS_INDIRECT=0

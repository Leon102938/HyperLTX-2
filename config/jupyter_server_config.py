c.ServerApp.tornado_settings = {
    "ws_ping_interval": 10000,
    "ws_ping_timeout": 30000,
}

c.ServerApp.terminado_settings = {
    "shell_command": ["/bin/bash", "-l"],
}
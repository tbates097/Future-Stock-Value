#!/bin/bash

SERVICE_NAME="com.portfolio.app"
PLIST_PATH="$HOME/Library/LaunchAgents/$SERVICE_NAME.plist"

case "$1" in
    start)
        echo "Starting Portfolio App..."
        launchctl load "$PLIST_PATH"
        echo "Portfolio App started. Access it at:"
        echo "Local machine: http://localhost:8050"
        ip=$(ipconfig getifaddr en0)
        if [ ! -z "$ip" ]; then
            echo "Other devices: http://$ip:8050"
        fi
        ;;
    stop)
        echo "Stopping Portfolio App..."
        launchctl unload "$PLIST_PATH"
        echo "Portfolio App stopped"
        ;;
    restart)
        echo "Restarting Portfolio App..."
        launchctl unload "$PLIST_PATH"
        sleep 2
        launchctl load "$PLIST_PATH"
        echo "Portfolio App restarted"
        ;;
    status)
        if launchctl list | grep -q "$SERVICE_NAME"; then
            echo "Portfolio App is running"
            echo "Access it at:"
            echo "Local machine: http://localhost:8050"
            ip=$(ipconfig getifaddr en0)
            if [ ! -z "$ip" ]; then
                echo "Other devices: http://$ip:8050"
            fi
        else
            echo "Portfolio App is not running"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac

exit 0

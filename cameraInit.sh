#!/bin/bash

# kill -9 $(ps aux | grep gphoto | tr -s ' ' | cut -d' ' -f2)

# gnome-terminal  -- bash -c "gphoto2 --stdout --set-config liveviewsize=0 --capture-movie | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 -s:v 1920x1280 -r 25 /dev/video0" &
# FOO_PID=$!
# echo ${FOO_PID}
# gphoto2 --stdout --set-config liveviewsize=0 --capture-movie | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 -s:v 1920x1280 -r 25 /dev/video0
# sleep 4
# ./build/depth_extraction_optimized
# echo ${FOO_PID}
# pkill -9 -t ${FOO_PID}
# ('gphoto2 --stdout --set-config liveviewsize=0 --capture-movie | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 -s:v 1920x1280 -r 25 /dev/video0'; './build/depth_extraction_optimized')






# First run gphoto2 and redirect output to /dev/null, this will allow the actual program to been shown
(gphoto2 --stdout --set-config liveviewsize=0 --capture-movie | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 -s:v 1920x1280 -r 25 /dev/video0)&> /dev/null &
FOO_PID=$!  # Store its PID

# Make sure that gphoto2 has been started
sleep 4
echo ${FOO_PID}
# Run the actual program
./build/depth_extraction_optimized

# Exit condition, kill gphoto2
# kill ${FOO_PID}
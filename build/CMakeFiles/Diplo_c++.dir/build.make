# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jim/Desktop/Linux_C_PP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jim/Desktop/Linux_C_PP/build

# Include any dependencies generated for this target.
include CMakeFiles/Diplo_c++.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Diplo_c++.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Diplo_c++.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Diplo_c++.dir/flags.make

CMakeFiles/Diplo_c++.dir/main.cpp.o: CMakeFiles/Diplo_c++.dir/flags.make
CMakeFiles/Diplo_c++.dir/main.cpp.o: ../main.cpp
CMakeFiles/Diplo_c++.dir/main.cpp.o: CMakeFiles/Diplo_c++.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jim/Desktop/Linux_C_PP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Diplo_c++.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Diplo_c++.dir/main.cpp.o -MF CMakeFiles/Diplo_c++.dir/main.cpp.o.d -o CMakeFiles/Diplo_c++.dir/main.cpp.o -c /home/jim/Desktop/Linux_C_PP/main.cpp

CMakeFiles/Diplo_c++.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Diplo_c++.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jim/Desktop/Linux_C_PP/main.cpp > CMakeFiles/Diplo_c++.dir/main.cpp.i

CMakeFiles/Diplo_c++.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Diplo_c++.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jim/Desktop/Linux_C_PP/main.cpp -o CMakeFiles/Diplo_c++.dir/main.cpp.s

CMakeFiles/Diplo_c++.dir/Time.cpp.o: CMakeFiles/Diplo_c++.dir/flags.make
CMakeFiles/Diplo_c++.dir/Time.cpp.o: ../Time.cpp
CMakeFiles/Diplo_c++.dir/Time.cpp.o: CMakeFiles/Diplo_c++.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jim/Desktop/Linux_C_PP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Diplo_c++.dir/Time.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Diplo_c++.dir/Time.cpp.o -MF CMakeFiles/Diplo_c++.dir/Time.cpp.o.d -o CMakeFiles/Diplo_c++.dir/Time.cpp.o -c /home/jim/Desktop/Linux_C_PP/Time.cpp

CMakeFiles/Diplo_c++.dir/Time.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Diplo_c++.dir/Time.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jim/Desktop/Linux_C_PP/Time.cpp > CMakeFiles/Diplo_c++.dir/Time.cpp.i

CMakeFiles/Diplo_c++.dir/Time.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Diplo_c++.dir/Time.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jim/Desktop/Linux_C_PP/Time.cpp -o CMakeFiles/Diplo_c++.dir/Time.cpp.s

# Object files for target Diplo_c++
Diplo_c_______OBJECTS = \
"CMakeFiles/Diplo_c++.dir/main.cpp.o" \
"CMakeFiles/Diplo_c++.dir/Time.cpp.o"

# External object files for target Diplo_c++
Diplo_c_______EXTERNAL_OBJECTS =

Diplo_c++: CMakeFiles/Diplo_c++.dir/main.cpp.o
Diplo_c++: CMakeFiles/Diplo_c++.dir/Time.cpp.o
Diplo_c++: CMakeFiles/Diplo_c++.dir/build.make
Diplo_c++: /usr/local/lib/libopencv_gapi.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_highgui.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_ml.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_objdetect.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_photo.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_stitching.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_video.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_videoio.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_imgcodecs.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_dnn.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_calib3d.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_features2d.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_flann.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_imgproc.so.4.6.0
Diplo_c++: /usr/local/lib/libopencv_core.so.4.6.0
Diplo_c++: CMakeFiles/Diplo_c++.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jim/Desktop/Linux_C_PP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable Diplo_c++"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Diplo_c++.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Diplo_c++.dir/build: Diplo_c++
.PHONY : CMakeFiles/Diplo_c++.dir/build

CMakeFiles/Diplo_c++.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Diplo_c++.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Diplo_c++.dir/clean

CMakeFiles/Diplo_c++.dir/depend:
	cd /home/jim/Desktop/Linux_C_PP/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jim/Desktop/Linux_C_PP /home/jim/Desktop/Linux_C_PP /home/jim/Desktop/Linux_C_PP/build /home/jim/Desktop/Linux_C_PP/build /home/jim/Desktop/Linux_C_PP/build/CMakeFiles/Diplo_c++.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Diplo_c++.dir/depend

